from torch import nn
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
import torch.nn.functional as F

from .utils import *

HW = 224 * 224 # image area
n_classes = 1000

def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

class CausalMetric():

    def __init__(self, model, mode, step, substrate_fn):
        r"""Create deletion/insertion metric instance.

        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn

    def single_run(self, img_tensor, explanation, verbose=0, save_to=None):
        r"""Run metric on one image-saliency pair.

        Args:
            img_tensor (Tensor): normalized image tensor.
            explanation (np.ndarray): saliency map.
            verbose (int): in [0, 1, 2].
                0 - return list of scores.
                1 - also plot final step.
                2 - also plot every step and print 2 top classes.
            save_to (str): directory to save every step plots to.

        Return:
            scores (nd.array): Array containing scores at every step.
        """
        pred = self.model(img_tensor.cuda())
        top, c = torch.max(pred, 1)
        c = c.cpu().numpy()[0]
        n_steps = (HW + self.step - 1) // self.step

        if self.mode == 'del':
            title = 'Deletion game'
            ylabel = 'Pixels deleted'
            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)
        elif self.mode == 'ins':
            title = 'Insertion game'
            ylabel = 'Pixels inserted'
            start = self.substrate_fn(img_tensor)
            
            finish = img_tensor.clone()

        scores = np.empty(n_steps + 1)
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(explanation.reshape(-1, HW), axis=1), axis=-1)
        for i in range(n_steps+1):
            pred = self.model(start.cuda())
            pr, cl = torch.topk(pred, 2)
            if verbose == 2:
                print('{}: {:.3f}'.format(get_class_name(cl[0][0]), float(pr[0][0])))
                print('{}: {:.3f}'.format(get_class_name(cl[0][1]), float(pr[0][1])))
            scores[i] = pred[0, c]
            # Render image if verbose, if it's the last step or if save is required.
            if verbose == 2 or (verbose == 1 and i == n_steps) or save_to:
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.title('{} {:.1f}%, P={:.4f}'.format(ylabel, 100 * i / n_steps, scores[i]))
                plt.axis('off')
                tensor_imshow(start[0])

                plt.subplot(122)
                plt.plot(np.arange(i+1) / n_steps, scores[:i+1])
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i+1) / n_steps, 0, scores[:i+1], alpha=0.4)
                plt.title(title)
                plt.xlabel(ylabel)
                plt.ylabel(get_class_name(c))
                if save_to:
                    plt.savefig(save_to + '/{:03d}.png'.format(i))
                    plt.close()
                else:
                    plt.show()
            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                start.cpu().numpy().reshape(1, 3, HW)[0, :, coords] = finish.cpu().numpy().reshape(1, 3, HW)[0, :, coords]
        return scores

    def evaluate(self, img_batch, exp_batch, batch_size):
        r"""Efficiently evaluate big batch of images.

        Args:
            img_batch (Tensor): batch of images.
            exp_batch (np.ndarray): batch of explanations.
            batch_size (int): number of images for one small batch.

        Returns:
            scores (nd.array): Array containing scores at every step for every image.
        """
        n_samples = img_batch.shape[0]
        predictions = torch.FloatTensor(n_samples, n_classes)
        assert n_samples % batch_size == 0
        for i in tqdm(range(n_samples // batch_size), desc='Predicting labels'):
            preds = self.model(img_batch[i*batch_size:(i+1)*batch_size].cuda()).cpu()
            predictions[i*batch_size:(i+1)*batch_size] = preds
        top = np.argmax(predictions, -1)
        n_steps = (HW + self.step - 1) // self.step
        scores = np.empty((n_steps + 1, n_samples))
        salient_order = np.flip(np.argsort(exp_batch.reshape(-1, HW), axis=1), axis=-1)
        r = np.arange(n_samples).reshape(n_samples, 1)

        substrate = torch.zeros_like(img_batch)
        for j in tqdm(range(n_samples // batch_size), desc='Substrate'):
            substrate[j*batch_size:(j+1)*batch_size] = self.substrate_fn(img_batch[j*batch_size:(j+1)*batch_size])

        if self.mode == 'del':
            caption = 'Deleting  '
            start = img_batch.clone()
            finish = substrate
        elif self.mode == 'ins':
            caption = 'Inserting '
            start = substrate
            finish = img_batch.clone()

        # While not all pixels are changed
        for i in tqdm(range(n_steps+1), desc=caption + 'pixels'):
            # Iterate over batches
            for j in range(n_samples // batch_size):
                # Compute new scores
                preds = self.model(start[j*batch_size:(j+1)*batch_size].cuda())
                preds = preds.cpu().numpy()[range(batch_size), top[j*batch_size:(j+1)*batch_size]]
                scores[i, j*batch_size:(j+1)*batch_size] = preds
            # Change specified number of most salient pixels to substrate pixels
            coords = salient_order[:, self.step * i:self.step * (i + 1)]
            start.cpu().numpy().reshape(n_samples, 3, HW)[r, :, coords] = finish.cpu().numpy().reshape(n_samples, 3, HW)[r, :, coords]
        print('AUC: {}'.format(auc(scores.mean(1))))
        return scores
    
    
class AdversarialCausalMetric(CausalMetric):
    def __init__(self, model, raw_model, mode, step, substrate_fn, hm_type, txt_embedding, txts, resize, preprocess):
        super().__init__(model, mode, step, substrate_fn)
        self.hm_type = hm_type
        self.txt_embedding = txt_embedding
        self.txts = txts
        self.resize = resize
        self.preprocess = preprocess
        self.raw_model = raw_model
        
    def single_run(self,
                   img_raw, # unnormalized image
                   explanation_fn,
                   target_class=None,
                   eps=32.0 / 255.0,
                   alpha=4.0 / 255.0,
                   pgd_steps=100,
                   deletion_steps=100,
                   process_batch_size=32,
                   deletion_batch_size=None,
                   clip_min=0.0,
                   clip_max=1.0,
                   return_details=True,
                   verbose=0):
        r"""PGD attack on a single image with classification + deletion loss.

        Args:
            img_raw (Tensor): unnormalized input image tensor with shape (1, C, H, W).
            explanation_fn (callable): function G that returns saliency for a given image.
            target_class (int, optional): class index c. If None, use model top-1 on clean image.
            eps (float): L_inf perturbation budget.
            alpha (float): PGD step size.
            pgd_steps (int): number of PGD iterations.
            deletion_steps (int): number of deletion iterations T.
            process_batch_size (int): number of process states evaluated per forward pass for both del/ins modes.
            deletion_batch_size (int, optional): deprecated alias of process_batch_size.
            margin (float): hinge margin m in deletion loss.
            lambda_del (float): weight for deletion loss.
            clip_min (float): minimum value for clamped normalized image.
            clip_max (float): maximum value for clamped normalized image.
            return_details (bool): whether to return optimization logs.
            verbose (int): if > 0, prints attack progress.

        Returns:
            Tensor or (Tensor, dict): adversarial image x_adv, and optional details.
        """
        if img_raw.shape[0] != 1:
            raise ValueError('AdversarialCausalMetric.single_run expects batch size 1.')

        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = img_raw.device

        x_raw = img_raw.detach().to(device) # unnormalized image
        with torch.no_grad():
            clean_logits = self.model(normalize_ImageNet1k(x_raw))
            if target_class is None:
                target_class = int(torch.argmax(clean_logits, dim=1).item())
        
        # clean_prob = clean_logits[0, target_class].item()

        delta = torch.zeros_like(x_raw, requires_grad=True)
        deletion_steps = (HW + self.step - 1) // self.step
        deletion_steps = int(max(1, deletion_steps))
        details = {
            'loss': [],
            'clean_prob': float(clean_logits[0, target_class].item()),
            'adv_prob': [],
            'pgd_trace': [],
        }
        best_x_raw_adv = None
        best_loss = None
        best_step = None

        if deletion_batch_size is not None:
            process_batch_size = deletion_batch_size

        for k in range(pgd_steps):
            x_raw_adv = torch.clamp(x_raw + delta, clip_min, clip_max)
            x_adv_normalzie = normalize_ImageNet1k(x_raw_adv)
            adv_logits = self.model(x_adv_normalzie)
            l_preserve = F.kl_div(
                adv_logits,
                clean_logits,
                reduction='batchmean',
            )
            # Ranking is treated as fixed in each PGD iteration.
            saliency = explanation_fn(
                self.raw_model, # clip_model (not including the softmax)
                self.hm_type, # edclip,gradcam
                x_adv_normalzie, # normalzied image
                self.txt_embedding,
                self.txts,
                self.resize,
                self.preprocess
            )
            if isinstance(saliency, torch.Tensor):
                saliency = saliency.detach().cpu().numpy()
            saliency = np.asarray(saliency)
            salient_order = np.flip(np.argsort(saliency.reshape(-1, HW), axis=1), axis=-1).copy()
            if self.mode == 'del':
                xt = x_raw_adv
                finish = self.substrate_fn(x_raw_adv)
            elif self.mode == 'ins':
                xt = self.substrate_fn(x_raw_adv)
                finish = x_raw_adv
            else:
                raise ValueError("mode must be 'del' or 'ins'")
            finish_flat = finish.view(1, 3, HW)
            xt_states = []


            for t in range(deletion_steps):
                xt_states.append(xt.clone())

                start_idx = self.step * t
                end_idx = min(HW, self.step * (t + 1))
                if start_idx >= HW:
                    break

                coords = torch.as_tensor(
                    salient_order[0, start_idx:end_idx],
                    device=device,
                    dtype=torch.long
                )

                xt_next = xt.clone()
                xt_next_flat = xt_next.view(1, 3, HW)
                xt_next_flat[0, :, coords] = finish_flat[0, :, coords]
                xt = xt_next

            if process_batch_size is None or process_batch_size < 1:
                process_batch_size = len(xt_states)

            l_del = torch.zeros(1, device=device)
            p_t_trace = []
            for start_idx in range(0, len(xt_states), process_batch_size):
                end_idx = min(len(xt_states), start_idx + process_batch_size)
                xt_batch = torch.cat(xt_states[start_idx:end_idx], dim=0)
                logits_batch = self.model(normalize_ImageNet1k(xt_batch))
                p_t_batch = logits_batch[:, target_class]
                l_del = l_del + p_t_batch.sum() # aggregation
                p_t_trace.extend([float(v) for v in p_t_batch.detach().cpu().tolist()])

            logits_last = self.model(normalize_ImageNet1k(xt)) # last logitss
            l_del += logits_last[:, target_class]
            l_del = l_del / deletion_steps # average deletion loss
            if self.mode == 'del':
                loss = l_del - l_preserve
            elif self.mode == 'ins':
                loss = l_del + l_preserve

            step_loss = float(loss.item())
            step_pred_label = int(torch.argmax(adv_logits, dim=1).item())
            class_preserved = step_pred_label == target_class
            if class_preserved and (best_loss is None or step_loss < best_loss):
            # if (best_loss is None or step_loss < best_loss):
                best_loss = step_loss
                best_x_raw_adv = x_raw_adv.detach().clone()
                best_step = k + 1
            
            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()

            with torch.no_grad():
                if self.mode == 'del':
                    delta += alpha * delta.grad.sign()
                else:
                    delta -= alpha * delta.grad.sign()
                delta.clamp_(-eps, eps)
            delta = delta.detach().requires_grad_(True)

            details['loss'].append(step_loss)
            details['pgd_trace'].append({
                'pgd_step': k + 1,
                'loss': step_loss,
                'class_preserved': bool(class_preserved),
                'p_t': p_t_trace,
            })

            if verbose:
                print('PGD {}/{} | L={:.6f}'.format(
                    k + 1,
                    pgd_steps,
                    details['loss'][-1]
                ))

        if best_x_raw_adv is None:
            x_raw_adv = torch.clamp(x_raw + delta.detach(), clip_min, clip_max)
            selected_step = pgd_steps
            selected_class_preserved = False
        else:
            x_raw_adv = best_x_raw_adv
            selected_step = best_step
            selected_class_preserved = True

        details['selected_step'] = int(selected_step)
        details['selected_class_preserved'] = bool(selected_class_preserved)
        details['selection_fallback_last_step'] = bool(best_x_raw_adv is None)
        details['selected_loss'] = float(
            best_loss
            if best_loss is not None
            else details['loss'][-1]
        )
        details['adv_prob'] = float(self.model(normalize_ImageNet1k(x_raw_adv))[0, target_class].item())

        if return_details:
            return x_raw_adv, details
        return x_raw_adv


class JointAdversarialCausalMetric(nn.Module):
    """Optimize deletion and insertion adversarial objectives jointly."""

    def __init__(
        self,
        model,
        raw_model,
        step,
        del_substrate_fn,
        ins_substrate_fn,
        hm_type,
        txt_embedding,
        txts,
        resize,
        preprocess,
    ):
        super().__init__()
        self.model = model
        self.raw_model = raw_model
        self.step = step
        self.del_substrate_fn = del_substrate_fn
        self.ins_substrate_fn = ins_substrate_fn
        self.hm_type = hm_type
        self.txt_embedding = txt_embedding
        self.txts = txts
        self.resize = resize
        self.preprocess = preprocess

    def _build_states(self, x_raw_adv, salient_order, run_mode, deletion_steps, device):
        if run_mode == "del":
            xt = x_raw_adv
            finish = self.del_substrate_fn(x_raw_adv)
        elif run_mode == "ins":
            xt = self.ins_substrate_fn(x_raw_adv)
            finish = x_raw_adv
        else:
            raise ValueError("run_mode must be 'del' or 'ins'")

        finish_flat = finish.view(1, 3, HW)
        xt_states = []
        for t in range(deletion_steps):
            xt_states.append(xt.clone())
            start_idx = self.step * t
            end_idx = min(HW, self.step * (t + 1))
            if start_idx >= HW:
                break

            coords = torch.as_tensor(
                salient_order[0, start_idx:end_idx],
                device=device,
                dtype=torch.long,
            )
            xt_next = xt.clone()
            xt_next_flat = xt_next.view(1, 3, HW)
            xt_next_flat[0, :, coords] = finish_flat[0, :, coords]
            xt = xt_next

        return xt_states, xt

    def _path_loss(self, x_raw_adv, salient_order, run_mode, target_class, deletion_steps, process_batch_size, device):
        xt_states, xt_last = self._build_states(x_raw_adv, salient_order, run_mode, deletion_steps, device)

        curr_batch_size = process_batch_size
        if curr_batch_size is None or curr_batch_size < 1:
            curr_batch_size = len(xt_states)

        loss_val = torch.zeros(1, device=device)
        p_t_trace = []
        for start_idx in range(0, len(xt_states), curr_batch_size):
            end_idx = min(len(xt_states), start_idx + curr_batch_size)
            xt_batch = torch.cat(xt_states[start_idx:end_idx], dim=0)
            logits_batch = self.model(normalize_ImageNet1k(xt_batch))
            p_t_batch = logits_batch[:, target_class]
            loss_val = loss_val + p_t_batch.sum()
            p_t_trace.extend([float(v) for v in p_t_batch.detach().cpu().tolist()])

        logits_last = self.model(normalize_ImageNet1k(xt_last))
        loss_val += logits_last[:, target_class]
        loss_val = loss_val / deletion_steps
        return loss_val, p_t_trace

    def single_run(
        self,
        img_raw,
        explanation_fn,
        target_class=None,
        eps=32.0 / 255.0,
        alpha=4.0 / 255.0,
        pgd_steps=100,
        deletion_steps=100,
        process_batch_size=32,
        deletion_batch_size=None,
        clip_min=0.0,
        clip_max=1.0,
        return_details=True,
        verbose=0,
    ):
        if img_raw.shape[0] != 1:
            raise ValueError("JointAdversarialCausalMetric.single_run expects batch size 1.")

        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = img_raw.device

        x_raw = img_raw.detach().to(device)
        with torch.no_grad():
            clean_logits = self.model(normalize_ImageNet1k(x_raw))
            if target_class is None:
                target_class = int(torch.argmax(clean_logits, dim=1).item())

        delta = torch.zeros_like(x_raw, requires_grad=True)
        deletion_steps = (HW + self.step - 1) // self.step
        deletion_steps = int(max(1, deletion_steps))

        if deletion_batch_size is not None:
            process_batch_size = deletion_batch_size

        details = {
            "loss": [],
            "clean_prob": float(clean_logits[0, target_class].item()),
            "adv_prob": [],
            "pgd_trace": [],
        }
        best_x_raw_adv = None
        best_loss = None
        best_step = None

        for k in range(pgd_steps):
            x_raw_adv = torch.clamp(x_raw + delta, clip_min, clip_max)
            x_adv_normalized = normalize_ImageNet1k(x_raw_adv)
            adv_logits = self.model(x_adv_normalized)
            l_preserve = F.kl_div(adv_logits, clean_logits, reduction="batchmean")

            saliency = explanation_fn(
                self.raw_model,
                self.hm_type,
                x_adv_normalized,
                self.txt_embedding,
                self.txts,
                self.resize,
                self.preprocess,
            )
            if isinstance(saliency, torch.Tensor):
                saliency = saliency.detach().cpu().numpy()
            saliency = np.asarray(saliency)
            salient_order = np.flip(np.argsort(saliency.reshape(-1, HW), axis=1), axis=-1).copy()

            l_del, p_t_trace_del = self._path_loss(
                x_raw_adv,
                salient_order,
                "del",
                target_class,
                deletion_steps,
                process_batch_size,
                device,
            )
            l_ins, p_t_trace_ins = self._path_loss(
                x_raw_adv,
                salient_order,
                "ins",
                target_class,
                deletion_steps,
                process_batch_size,
                device,
            )

            objective_del = l_del - l_preserve
            objective_ins = -(l_ins + l_preserve) # twice the weight of l_preserve
            # objective_ins = -l_ins
            loss = objective_del + objective_ins

            grad_del = torch.autograd.grad(objective_del, delta, retain_graph=True, allow_unused=True)[0]
            grad_ins = torch.autograd.grad(objective_ins, delta, retain_graph=True, allow_unused=True)[0]
            if grad_del is None or grad_ins is None:
                grad_cosine = float("nan")
                grad_dot = float("nan")
                grad_norm_del = float("nan")
                grad_norm_ins = float("nan")
                grad_norm_total = float("nan")
                grad_cancellation_ratio = float("nan")
            else:
                grad_del_flat = grad_del.reshape(-1)
                grad_ins_flat = grad_ins.reshape(-1)
                grad_dot_tensor = torch.dot(grad_del_flat, grad_ins_flat)
                grad_norm_del_tensor = torch.norm(grad_del_flat, p=2)
                grad_norm_ins_tensor = torch.norm(grad_ins_flat, p=2)
                grad_norm_total_tensor = torch.norm((grad_del_flat + grad_ins_flat), p=2)

                denom = grad_norm_del_tensor * grad_norm_ins_tensor
                if float(denom.item()) <= 1e-12:
                    grad_cosine = float("nan")
                else:
                    grad_cosine = float((grad_dot_tensor / denom).item())

                grad_dot = float(grad_dot_tensor.item())
                grad_norm_del = float(grad_norm_del_tensor.item())
                grad_norm_ins = float(grad_norm_ins_tensor.item())
                grad_norm_total = float(grad_norm_total_tensor.item())
                grad_cancellation_ratio = float(
                    grad_norm_total_tensor.item() / (grad_norm_del + grad_norm_ins + 1e-12)
                )

            step_loss = float(loss.item())
            step_pred_label = int(torch.argmax(adv_logits, dim=1).item())
            class_preserved = step_pred_label == target_class
            if class_preserved and (best_loss is None or step_loss < best_loss):
                best_loss = step_loss
                best_x_raw_adv = x_raw_adv.detach().clone()
                best_step = k + 1

            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()

            with torch.no_grad():
                delta += alpha * delta.grad.sign()
                delta.clamp_(-eps, eps)
            delta = delta.detach().requires_grad_(True)

            details["loss"].append(step_loss)
            details["pgd_trace"].append(
                {
                    "pgd_step": k + 1,
                    "loss": step_loss,
                    "loss_del": float(l_del.item()),
                    "loss_ins": float(l_ins.item()),
                    "grad_cosine_l1_l2": grad_cosine,
                    "grad_dot_l1_l2": grad_dot,
                    "grad_norm_l1": grad_norm_del,
                    "grad_norm_l2": grad_norm_ins,
                    "grad_norm_l1_plus_l2": grad_norm_total,
                    "grad_cancellation_ratio": grad_cancellation_ratio,
                    "class_preserved": bool(class_preserved),
                    "p_t": {
                        "del": p_t_trace_del,
                        "ins": p_t_trace_ins,
                    },
                }
            )

            if verbose:
                print(
                    "PGD {}/{} | L={:.6f} | L_del={:.6f} | L_ins={:.6f}".format(
                        k + 1,
                        pgd_steps,
                        details["loss"][-1],
                        float(l_del.item()),
                        float(l_ins.item()),
                    )
                )

        if best_x_raw_adv is None:
            x_raw_adv = torch.clamp(x_raw + delta.detach(), clip_min, clip_max)
            selected_step = pgd_steps
            selected_class_preserved = False
        else:
            x_raw_adv = best_x_raw_adv
            selected_step = best_step
            selected_class_preserved = True

        details["selected_step"] = int(selected_step)
        details["selected_class_preserved"] = bool(selected_class_preserved)
        details["selection_fallback_last_step"] = bool(best_x_raw_adv is None)
        details["selected_loss"] = float(
            best_loss
            if best_loss is not None
            else details["loss"][-1]
        )
        details["adv_prob"] = float(self.model(normalize_ImageNet1k(x_raw_adv))[0, target_class].item())

        if return_details:
            return x_raw_adv, details
        return x_raw_adv
