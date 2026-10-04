import torch
import torch.optim as optim

from accuracy import calculate_class_accuracy
from models import resnet18, classifier
from torch.nn import functional as F
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pro_loss(f, y, pro, t=1.0):
    num_pro = pro.shape[0]
    if num_pro == 0:
        return torch.tensor(0.0, device=f.device, dtype=f.dtype)

    pro_tensor = pro.to(device)
    batch_size = f.shape[0]

    f_expand = f.unsqueeze(1)  # [batch, 1, feature_dim]
    pro_expand = pro_tensor.unsqueeze(0)  # [1, num_pro, feature_dim]
    distances = torch.norm(f_expand - pro_expand, p=2, dim=2)  # [batch, num_pro]

    y_expand = y.unsqueeze(1)  # [batch, 1]
    pro_indices = torch.arange(num_pro).unsqueeze(0).to(f.device)  # [1, num_pro]
    same_label_mask = (y_expand == pro_indices).float()

    same_loss = distances * same_label_mask
    diff_loss = torch.clamp(t - distances, min=0.0) * (1 - same_label_mask)

    total_loss = same_loss + diff_loss
    avg_loss = total_loss.sum() / batch_size

    return avg_loss


def w_loss(f, y, w, t=0.5):
    num_w = w.shape[0]
    if num_w == 0:
        return torch.tensor(0.0, device=device, dtype=f.dtype)

    w_tensor = w.to(device)
    batch_size = f.shape[0]

    cos_sim = torch.matmul(f, w_tensor.T)

    y_expand = y.unsqueeze(1)  # [batch, 1]
    w_indices = torch.arange(num_w).unsqueeze(0).to(f.device)
    same_label_mask = (y_expand == w_indices).float()

    same_loss = (1 - cos_sim) * same_label_mask

    diff_loss = torch.clamp(cos_sim - t, min=0.0) * (1 - same_label_mask)

    total_loss = same_loss + diff_loss
    avg_loss = total_loss.sum() / batch_size

    return avg_loss


def compute_avg_d(features, labels, prototypes, num_classes):
    """
    input:
        features: [batchsize, 2048]
        labels: [batchsize]
        prototypes: [numclasses, 2048]

    output:
    """
    avg_cos = torch.zeros(num_classes, device=device, requires_grad=False)
    avg_d = torch.zeros(num_classes, device=device, requires_grad=False)

    for cls in range(num_classes):
        cls_mask = (labels == cls)
        cls_feats = features[cls_mask]
        n_samples = cls_feats.shape[0]
        if n_samples == 0:
            avg_cos[cls] = 0.0
            continue

        cls_proto = prototypes[cls]
        cos_sim = F.cosine_similarity(cls_feats, cls_proto, dim=1)
        # [-1, 1] -> [0,1]
        norm_cos = (cos_sim + 1.0) / 2.0
        norm_d = torch.sqrt(2 - 2 * norm_cos)
        avg_cos[cls] = torch.mean(norm_cos)
        avg_d[cls] = torch.mean(norm_d)

    return avg_cos, avg_d


def sim_data(f, y, num_classes, num):
    label2indices = {}
    for label in torch.unique(y):
        label = label.item()
        label2indices[label] = torch.where(y == label)[0].tolist()

    if len(label2indices) < 2:
        raise ValueError("error")

    sim_f_list = []
    sim_soft_label_list = []
    sim_hard_label = torch.full((num,), num_classes, dtype=torch.long, device=device)

    feature_dim = f.shape[1]
    half_dim = feature_dim // 2
    all_labels = list(label2indices.keys())

    for _ in range(num):
        c1, c2 = random.sample(all_labels, 2)

        idx1 = random.choice(label2indices[c1])
        idx2 = random.choice(label2indices[c2])
        f1 = f[idx1]
        f2 = f[idx2]

        sim_f = torch.cat([f1[:half_dim], f2[half_dim:]], dim=0)
        sim_f_list.append(sim_f)

        sim_soft = torch.zeros(num_classes, dtype=torch.float32, device=device)
        sim_soft[c1] = 0.5
        sim_soft[c2] = 0.5
        sim_soft_label_list.append(sim_soft)

    sim_f = torch.stack(sim_f_list, dim=0)  # [num, 512]
    sim_soft_label = torch.stack(sim_soft_label_list, dim=0)  # [num, num_classes]

    return sim_f, sim_hard_label, sim_soft_label


def neighbor(features, pred_logits, prototypes, n, num_classes):
    num_classes = prototypes.shape[0]
    dist_matrix = torch.cdist(features, prototypes)  # shape [B, num_classes]
    min_dist, pseudo_labels = torch.min(dist_matrix, dim=1)

    selected_pred_list = []
    selected_plist = []

    for cls in range(num_classes):
        cls_mask = pseudo_labels == cls
        cls_pred = pred_logits[cls_mask]
        cls_dists = min_dist[cls_mask]

        if len(cls_pred) == 0:
            continue

        sorted_idx = torch.argsort(cls_dists)
        take_n = min(n, len(sorted_idx))
        topn_idx = sorted_idx[:take_n]

        selected_pred_list.append(cls_pred[topn_idx])
        selected_plist.append(torch.full((take_n,), cls, dtype=torch.long))

    selected_pred = torch.cat(selected_pred_list, dim=0)
    selected_pseudo_labels = torch.cat(selected_plist, dim=0)

    return selected_pred, selected_pseudo_labels


def compute_custom_loss(pred_logits, threshold_t):
    alpha_i, q = torch.max(pred_logits, dim=1)

    h_t = threshold_t[q]

    numerator = torch.square(alpha_i - h_t)

    # Step4 min{α_i*(1-α_i), h_t*(1-h_t)}
    term1 = alpha_i * (1.0 - alpha_i)
    term2 = h_t * (1.0 - h_t)
    denominator = torch.min(term1, term2)

    denominator = denominator + 1e-8

    per_sample = numerator / denominator

    # 1/n_T * sum()
    loss = torch.mean(per_sample)

    return loss


def calculate_os_uk_h(true_labels, pred_labels, num_classes):
    gt_known_mask = true_labels < num_classes
    gt_unknown_mask = true_labels == num_classes

    gt_known_correct = torch.logical_and(gt_known_mask, pred_labels == true_labels)
    count_gt_known_correct = torch.sum(gt_known_correct).float()
    count_total_gt_known = torch.sum(gt_known_mask).float()

    if count_total_gt_known > 1e-8:
        OS = count_gt_known_correct / count_total_gt_known
    else:
        OS = torch.tensor(0.0)

    gt_unknown_correct = torch.logical_and(gt_unknown_mask, pred_labels == num_classes)
    count_gt_unknown_correct = torch.sum(gt_unknown_correct).float()
    count_total_gt_unknown = torch.sum(gt_unknown_mask).float()

    if count_total_gt_unknown > 1e-8:
        UK = count_gt_unknown_correct / count_total_gt_unknown
    else:
        UK = torch.tensor(0.0)

    OS = OS.item()
    UK = UK.item()
    if OS + UK < 1e-8:
        H = 0.0
    else:
        H = (2 * OS * UK) / (OS + UK)

    return OS, UK, H


def get_final_labels(pred_logits, features, prototypes, threshold_t, num_classes):
    _, q = torch.max(pred_logits, dim=1)

    q_protos = prototypes[q]

    dist = torch.norm(features - q_protos, p=2, dim=1)

    h_t = threshold_t[q]  # [B]

    mask = dist < h_t
    final_labels = torch.where(mask, q, torch.full_like(q, fill_value=num_classes))

    return final_labels


def pre_training(ds, dst, num_classes, epochs=100):
    Gf = resnet18().to(device)
    Gc = classifier(512, num_classes).to(device)

    parameter_list = [{"params": Gf.parameters()}, {"params": Gc.parameters()}]
    optimizer = optim.SGD(parameter_list, lr=0.001, momentum=0.9, weight_decay=0.0005)

    hcos = torch.zeros(num_classes, device=device, requires_grad=False)
    hd = torch.zeros(num_classes, device=device, requires_grad=False)

    for epoch in range(epochs):

        lc = lcon = lcos = slc = slcon = slcos = 0.0
        cacc = [0.0] * (num_classes + 1)

        t = torch.zeros(num_classes, device=device, requires_grad=False)

        Gf.train()
        Gc.train()

        for (x, y) in ds:

            sum_f = torch.zeros(num_classes, 512).to(device)
            sum_p = torch.zeros(num_classes).to(device)

            x = x.to(device)
            y = y.to(device)
            f = Gf(x).squeeze(-1)
            f = F.normalize(f, p=2, dim=1, eps=1e-12)
            out = Gc(f)

            p = F.softmax(out, dim=1)[range(len(y)), y]
            for c in range(num_classes):
                mask = (y == c)
                if mask.sum() > 0:
                    sum_f[c] += (f[mask] * p[mask].unsqueeze(1)).sum(0)
                    sum_p[c] += p[mask].sum()
            sum_p_safe = torch.clamp(sum_p, min=1e-12).unsqueeze(1)
            pro = sum_f / sum_p_safe
            pro = F.normalize(pro, p=2, dim=1, eps=1e-12)

            w = Gc.fc[1].weight
            w = F.normalize(w, p=2, dim=1, eps=1e-12)

            y_onehot = F.one_hot(y, num_classes=num_classes).float().to(device)
            out_softmax = F.softmax(out, dim=1) + 1e-12

            sx, sy, soy = sim_data(f, y, num_classes, 32)
            sout = F.softmax(Gc(sx), dim=1) + 1e-12

            lcl = -torch.sum(y_onehot * torch.log(out_softmax), dim=1).mean()
            lconl = pro_loss(f, y, pro)
            lcosl = w_loss(f, y, w)
            slcl = -torch.sum(soy * torch.log(sout), dim=1).mean()
            slconl = pro_loss(sx, sy, pro)
            slcosl = w_loss(sx, sy, w)

            loss = (1 / 6) * (lcl + lconl + lcosl + slcl + slconl + slcosl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lc += lcl / len(ds)
            lcon += lconl / len(ds)
            lcos += lcosl / len(ds)
            slc += slcl / len(ds)
            slcon += slconl / len(ds)
            slcos += slcosl / len(ds)

        Gf.eval()
        Gc.eval()

        for (x, y) in dst:

            x = x.to(device)
            y = y.to(device)
            f = Gf(x).squeeze(-1)
            f = F.normalize(f, p=2, dim=1, eps=1e-12)
            out = Gc(f)

            acc = calculate_class_accuracy(out, y)
            for i in range(len(cacc)):
                cacc[i] += (acc[i] / len(dst))

        print(f"{epoch + 1} / {epochs}")
        print(f"lc : {lc:.3f}",
              f"lcon : {lcon:.3f}",
              f"lcos : {lcos:.3f}",
              f"slc : {slc:.3f}",
              f"slcon : {slcon:.3f}",
              f"slcos : {slcos:.3f}", )
        for i in range(len(cacc)):
            print(f"{i} : {cacc[i]:.3f}")
        print()

    Gf.eval()
    Gc.eval()

    for (x, y) in ds:
        x = x.to(device)
        y = y.to(device)
        f = Gf(x).squeeze(-1)
        f = F.normalize(f, p=2, dim=1, eps=1e-12)

        w = Gc.fc[1].weight
        w = F.normalize(w, p=2, dim=1, eps=1e-12)

        avg_cos, avg_d = compute_avg_d(f, y, w, num_classes)
        hcos += avg_cos / len(ds)
        hd += avg_d / len(ds)

    torch.save(Gf.state_dict(), 'training/model/Gf.pth')
    torch.save(Gc.state_dict(), 'training/model/Gc.pth')

    return hcos.detach(), hd.detach()


def fine_tuning(ds, num_classes, hcos, epochs=200):
    Gf = resnet18().to(device)
    Gc = classifier(512, num_classes).to(device)
    Gf.load_state_dict(torch.load('training/model/Gf.pth', weights_only=True))
    Gc.load_state_dict(torch.load('training/model/Gc.pth', weights_only=True))

    parameter_list = [{"params": Gf.parameters()}]
    optimizer = optim.SGD(parameter_list, lr=0.001, momentum=0.9, weight_decay=0.0005)

    for epoch in range(epochs):

        lc = lt = 0.0

        Gf.train()
        Gc.eval()

        for (x, _) in ds:
            x = x.to(device)
            f = Gf(x).squeeze(-1)
            f = F.normalize(f, p=2, dim=1, eps=1e-12)
            out = Gc(f)
            out = F.softmax(out, dim=1) + 1e-12

            w = Gc.fc[1].weight
            w = F.normalize(w, p=2, dim=1, eps=1e-12)

            out2, y2 = neighbor(f, out, w, 3, num_classes)

            y_onehot = F.one_hot(y2, num_classes=num_classes).float().to(device)

            lcl = -torch.sum(y_onehot * torch.log(out2), dim=1).mean()

            ltl = compute_custom_loss(out, hcos)

            loss = 0.5 * lcl + 0.5 * ltl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lc += lcl / len(ds)
            lt += ltl / len(ds)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"lc={lc:.3f}, lt={lt:.3f}")
        print()

    torch.save(Gf.state_dict(), 'training/model/Gf2.pth')
    torch.save(Gc.state_dict(), 'training/model/Gc2.pth')


def test(dst, hd, num_classes):

    Gf = resnet18().to(device)
    Gc = classifier(512, num_classes).to(device)
    Gf.load_state_dict(torch.load('training/model/Gf2.pth', weights_only=True))
    Gc.load_state_dict(torch.load('training/model/Gc2.pth', weights_only=True))

    Gf.eval()
    Gc.eval()

    all_f = []
    all_out = []
    all_y = []

    for (x, y) in dst:
        x = x.to(device)
        y = y.to(device)
        f = Gf(x).squeeze(-1)
        f = F.normalize(f, p=2, dim=1, eps=1e-12)
        out = Gc(f)
        out = torch.softmax(out, dim=1)

        all_f.append(f.detach().cpu())
        all_out.append(out.detach().cpu())
        all_y.append(y.detach().cpu())

    all_f = torch.cat(all_f, dim=0)
    all_out = torch.cat(all_out, dim=0)
    all_y = torch.cat(all_y, dim=0)

    w = Gc.fc[1].weight.cpu()
    w = F.normalize(w, p=2, dim=1, eps=1e-12)
    hd = hd.cpu()

    y_pre = get_final_labels(all_out, all_f, w, hd, num_classes)

    OS, UK, H = calculate_os_uk_h(all_y, y_pre, num_classes)
    print(f"OS={OS:.2f}, UK={UK:.2f}, H={H:.2f}")
