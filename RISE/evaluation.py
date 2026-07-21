from torch import nn
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter

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

        delta = torch.zeros_like(x_raw, requires_grad=True)
        deletion_steps = (HW + self.step - 1) // self.step
        deletion_steps = int(max(1, deletion_steps))
        details = {
            'loss': [],
            'loss_cls': [],
            'loss_del': []
        }

        for k in range(pgd_steps):
            x_raw_adv = torch.clamp(x_raw + delta, clip_min, clip_max)
            x_adv_normalzie = normalize_ImageNet1k(x_raw_adv)
    
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
            l_del = torch.zeros(1, device=device)

            for t in range(deletion_steps):
                logits_t = self.model(xt)
                p_t = logits_t[:, target_class]
                l_del = l_del + p_t # aggregation

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

            logits_last = self.model(xt) # last logitss
            l_del += logits_last[:, target_class]
            loss = l_del / deletion_steps # average deletion loss

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

            details['loss'].append(float(loss.item()))

            if verbose:
                print('PGD {}/{} | L={:.6f}'.format(
                    k + 1,
                    pgd_steps,
                    details['loss'][-1]
                ))

        x_raw_adv = torch.clamp(x_raw + delta.detach(), clip_min, clip_max)
        if return_details:
            return x_raw_adv, details
        return x_raw_adv
