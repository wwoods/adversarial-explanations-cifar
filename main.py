"""Main file for training / explaining ResNet-44 networks on the CIFAR-10
dataset.
"""

### Local imports
import model

### Environment imports
import click
import math
import os
import shutil
import torch
import torch.utils.data.dataset
import torchvision

### Configuration settings
# Training rate / size parameters
TRAIN_BATCHSIZE = 128  # Multiplied by number of GPUs
TRAIN_LR = 0.1  # Multiplied by number of GPUs
TRAIN_MOMENTUM = 0.9
TRAIN_WEIGHT_DECAY = 1e-4
TRAIN_EPOCHS = [170, 195, 200]  # Divide lr by 10 at each; finish after last.

# Adversarial training settings
TRAIN_ADV_EPS = 0.01
TRAIN_ADV_L2MIN_EPS = 0.1

# Adversarial robustness parameters
ROBUST_Z = 2
ROBUST_ZETA = 0.2  # Always tandem
ROBUST_ADAPT_L_TARGET = 1.5
ROBUST_ADAPT_PSI_0 = 220
ROBUST_ADAPT_PSI = 0.02
ROBUST_ADAPT_EPS_POS = 1
ROBUST_ADAPT_EPS_NEG = 0.01

# Offset as [mean, std] of data input.
MODEL_INPUT_OFFSET = [[0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]]

# Network architecture
def cifar10_preprocess(ft_out):
    return torch.nn.Conv2d(3, ft_out, kernel_size=3, padding=1, bias=False)
MODEL_ARCH = [
        32,  # Input size; assumed square
        cifar10_preprocess,  # Initial layer(s) to ResNet
        [(44 - 2)//6 for _ in range(3)],  # Block lengths
        [16, 32, 64],  # Number of features
        10,  # Number of classes
]

# For testing on low-resource computers
ONE_BATCH_ONLY = False


### Set up dataset

# Find CIFAR-10
cifar10_path = os.environ.get('CIFAR10_PATH', '')
if not cifar10_path.strip():
    raise ValueError('Must specify CIFAR10_PATH environment variable.')

# Labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
        'frog', 'horse', 'ship', 'truck']


### Commandline functions

@click.group()
def main():
    """Command which trains/explains ResNet-44 networks on the CIFAR-10
    dataset.
    """
    # Ensure CIFAR-10 is downloaded
    if not os.path.lexists(cifar10_path):
        torchvision.datasets.CIFAR10(cifar10_path, download=True)


@main.command()
@click.argument('path')
@click.option('--n-images', default=1000, type=int)
@click.option('--eps', default=20, type=float)
@click.option('--steps', default=450, type=int)
@click.option('--momentum', default=0.9, type=float)
def calculate_ara(path, n_images, eps, steps, momentum):
    """Calculates the Attack and BTR Accuracy-Robustness Areas (ARAs) for
    PATH.

    Options:

        --n-images: Number of images to use for the ARA calculation.  Defaults
                to 1000.

        --eps: Maximum perturbation to apply.  May be greater than 1; per
                Algorithm 1, many of the steps taken will minimize the
                perturbation magnitude instead of decreasing confidence in the
                correct class.  Defaults to 450*.1*.5 = 20, which is about what
                the paper used.

        --steps: Number of steps to use in determining the lowest-magnitude
                perturbation which results in a change of either accuracy or
                BTR, based on the ARA being calculated.  Defaults to 450.

        --momentum: If using fewer than 100 ``steps``, probably best to disable
                momentum by setting it to zero.  Defaults to 0.9.

    """
    # Set up model
    m = _model_load(path)

    device = torch.device('cpu')
    if torch.cuda.device_count() > 0:
        device = torch.device('cuda')
        m = m.to(device)
        m = torch.nn.DataParallel(m)

    m.eval()

    # Set up dataset based on first --n-images of random test-set permutation
    ds_train, ds_test = _get_dataset()
    state = torch.get_rng_state()
    torch.manual_seed(1776)
    idx_all = [int(i) for i in torch.randperm(len(ds_test))]
    torch.set_rng_state(state)
    ds_test = torch.utils.data.dataset.Subset(ds_test, idx_all[:n_images])

    for ara_name, ara_type in [('Attack', True), ('BTR', 'btr')]:
        batch_size = TRAIN_BATCHSIZE * max(1, torch.cuda.device_count())
        test_loader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size,
                shuffle=False, num_workers=8, drop_last=False)

        n = 0
        diffs = []
        for batch in test_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            n += images.size(0)

            advs = _adv_images(m, images, labels, AdversarialOptions(
                    steps=steps, eps=eps, eps_overshoot=1, use_l2min=ara_type,
                    momentum=momentum))
            mags = advs.sub_(images).pow_(2).mean((1, 2, 3)).sqrt_()
            diffs.extend(mags.tolist())
            print('.', end='', flush=True)
        print('')
        assert n == n_images
        assert n == len(diffs)

        # Convert from perturbations to percentages
        naive_guess = 1. / m.module.training_options['arch'][-1]
        diffs = torch.Tensor(diffs)
        bins = 1000
        diff_max = diffs.max().item()
        hist = diffs.histc(bins, min=0, max=diff_max)
        gap = diff_max / bins
        ara = gap * (1 - hist.cumsum(0) / n_images).add_(-naive_guess).clamp_(min=0).sum()
        print(f'{ara_name} ARA: {ara}')


@main.command()
@click.argument('path')
@click.option('--eps', default=0.1)
def explain(path, eps):
    """Explains the first 10 testing examples from the CIFAR-10 dataset.

    Stores results in 'output/'.
    """
    output_dir = 'output'
    try:
        shutil.rmtree(output_dir)
    except FileNotFoundError:
        pass
    ds_train, ds_test = _get_dataset()
    m = _model_load(path)

    # Determine cuda status
    device = torch.device('cpu')
    if torch.cuda.device_count() > 0:
        device = torch.device('cuda')
        m = m.to(device)
        # Multi-GPU slower with small-batch explanations
        #m = torch.nn.DataParallel(m)

    # Generate explanations
    m.eval()
    for i in range(10):
        img, label = ds_test[i]
        img = img.to(device)

        name = f'{i}-{class_names[label]}'
        print(name)
        d = os.path.join(output_dir, name)
        os.makedirs(d)

        # save_image changes the image with torchvision v0.2.2
        torchvision.utils.save_image(img.clone(),
                os.path.join(d, '_input.png'))

        with torch.no_grad():
            guesses = m(img.unsqueeze(0))[0]
        sm = torch.nn.functional.softmax(guesses, 0)
        second_g = guesses.clone()
        second_g[label] = second_g.min() - 1
        second = second_g.argmax(0).item()

        targs = [label, second, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        expls = _adv_images(m, img.unsqueeze(0).repeat(len(targs), 1, 1, 1),
                targs, AdversarialOptions(encourage_labels=True,
                    eps_overshoot=5, eps=eps, steps=35))

        torchvision.utils.save_image(expls[0], os.path.join(d,
                f'_real_was_{sm[label]:.3f}.png'))
        torchvision.utils.save_image(expls[1], os.path.join(d,
                f'_second_{class_names[second]}_was_{sm[second]:.3f}.png'))
        for targ, expl in zip(targs[2:], expls[2:]):
            torchvision.utils.save_image(expl, os.path.join(d,
                f'{targ}_{class_names[targ]}_was_{sm[targ]:.3f}.png'))


@main.command()
@click.argument('path')
@click.option('--adversarial-training/--no-adversarial-training', default=False)
@click.option('--l2-min/--no-l2-min', default=False)
@click.option('--robust-additions/--no-robust-additions', default=False)
def train(path, **training_options):
    r"""Trains a network and saves the result at PATH.

    Options:

        --adversarial-training: Train the network using adversarial examples.
                By default, the adversarial examples are generated using a
                standard "L_2" loss function and an epsilon of 0.01.

        --l2-min: Only valid with --adversarial-training.  If specified, use
                the "L_{2,min}"  method of generating adversarial examples,
                with an epsilon of 0.1.

        --robust-additions: Train with the best settings of the other
                modifications in the paper, including: defense via Lipschitz
                Continuity with "L_{adv,z=2}", "\zeta = 0.2" using
                "L_{adv,tandem}", the Half-Huber ReLU, no output zeroing,
                an adaptive "\psi" with "L_{target}=1.5", "k_{\psi,0}=220",
                "k_{\psi}=\ln 0.02", "\epsilon_{better}=1",
                "\epsilon_{worse}=0.01", and half-half adversarial training when
                also using adversarial training.
    """
    ds_train, ds_test = _get_dataset()
    training_options['arch'] = MODEL_ARCH
    training_options['input_offset'] = MODEL_INPUT_OFFSET
    m = model.Model(training_options)

    # Determine number of GPUs, update training batch size and learn rate
    g = max(torch.cuda.device_count(), 1)
    batch_size = TRAIN_BATCHSIZE * g
    opt = torch.optim.SGD(m.parameters(), lr=TRAIN_LR * g,
            momentum=TRAIN_MOMENTUM, weight_decay=TRAIN_WEIGHT_DECAY)

    # Move to GPU if needed
    device = torch.device('cpu')
    m_orig = m  # Preserve original reference for saving
    if torch.cuda.device_count() > 0:
        device = torch.device('cuda')
        m = m.to(device)
        m = torch.nn.DataParallel(m)

    # Training loop
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size,
            shuffle=True, num_workers=8, drop_last=False)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size,
            shuffle=False, num_workers=8, drop_last=False)
    for epoch in range(max(TRAIN_EPOCHS)):
        print(f'== {epoch} ==')
        # Update learning rate
        lr = TRAIN_LR * g
        for e in TRAIN_EPOCHS:
            if epoch >= e:
                lr *= 0.1
        for pg in opt.param_groups:
            pg['lr'] = lr

        # Train on batches
        m.train()
        n = 0
        loss = 0.
        for batch in train_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            n += images.size(0)

            if training_options['adversarial_training']:
                adv_opts = AdversarialOptions(
                        use_l2min=training_options['l2_min'],
                        eps=0.1 if training_options['l2_min'] else 0.01,
                        use_half_and_half=training_options['robust_additions'],
                        ensure_proper_minimization=False,
                )
                images = _adv_images(m, images, labels, adv_opts)

            stats = _model_train_batch(m, training_options, opt, images,
                    labels)
            loss += stats['class_loss_sum']
            if ONE_BATCH_ONLY:
                break
        print(f'Loss: {loss / n:.4f}')
        if training_options['robust_additions']:
            print(f'Psi: {_model_robust_get_psi(training_options["robust_integration"]):.1f}')

        _model_save(path, m_orig, training_options)

    # Done; test
    stats = _model_evaluate(m, test_loader)
    print("Testing statistics:")
    for k, v in stats.items():
        print(f'{k}: {v}')


@main.command()
def github_copy_example_output():
    """Administrative command; copies files from output/0-cat to example-output,
    with appropriate renames.
    """
    if not os.path.lexists('example_output'):
        os.makedirs('example_output')
    pdir = os.path.join('output', '0-cat')
    for f in os.listdir(pdir):
        if '_second_' in f:
            fnew = f.split('_second_')[0] + '_second.png'
        elif '_was_' in f:
            fnew = f.split('_was_')[0] + '.png'
        else:
            fnew = f
        shutil.copy2(os.path.join(pdir, f),
                os.path.join('example_output', fnew))


@main.command()
def github_prebuilt_images():
    """Administrative command.  Creates example_output/prebuilt*.png.

    Based on 8-cat, 9-automobile, which seem interesting.
    """
    p = 'example_output'
    if not os.path.lexists(p):
        os.makedirs(p)
    for f in os.listdir(p):
        if f.startswith('prebuilt'):
            os.unlink(os.path.join(p, f))

    ds_train, ds_test = _get_dataset()
    models = ['resnet44-standard.pt', 'resnet44-adv-train.pt',
            'resnet44-all.pt', 'resnet44-robust.pt']
    for i, m_file in enumerate(models):
        m = _model_load(os.path.join('prebuilt', m_file))
        device = torch.device('cpu' if torch.cuda.device_count() == 0
                else 'cuda')
        m = m.to(device)
        m.eval()
        imgs = []
        labels = []
        for img_id in [2, 5, 8, 9]:
            img, label = ds_test[img_id]
            imgs.append(img)
            labels.append(label)
        imgs = torch.stack(imgs).to(device)

        # Targets: real
        expls = _adv_images(m, imgs, labels, AdversarialOptions(
                encourage_labels=True, eps_overshoot=5, eps=0.1, steps=35))

        for j in range(imgs.size(0)):
            if i == 0:
                # Save original
                torchvision.utils.save_image(imgs[j], os.path.join(p,
                        f'prebuilt_orig_{j}.png'))
            torchvision.utils.save_image(expls[j], os.path.join(p,
                    f'prebuilt_{i}_{j}.png'))


### Implementation / helper functions

class AdversarialOptions:
    encourage_labels = False  # True for explanations
    eps = 0.1
    eps_overshoot = 1.  # Multiplier for step size; if > 1, uses g_explain
    momentum = 0  # Momentum for generating the explanations; with a large
                  # step count, this is useful.  Otherwise not very useful.
    steps = 7
    use_half_and_half = False
    use_l2min = False
    ensure_proper_minimization = True
    # For explanations and ARA calculations, ensure_proper_minimization serves
    # as a sanity check.  Basically, it checks that an adversarial example
    # satisfying the criteria is found prior to half the number of optimization
    # steps taken, ensuring that some number of the remaining steps are used
    # for minimizing the perturbation.

    def __init__(self, **kw):
        for k, v in kw.items():
            if not hasattr(AdversarialOptions, k):
                raise ValueError(k)
            setattr(self, k, v)


def _adv_images(m, images, labels, opts):
    if opts.steps <= 0:
        return images

    images = images.detach()
    deltas = images.new_zeros(images.size()).requires_grad_()

    if opts.use_half_and_half:
        affected = torch.rand(images.size(0), 1, 1, 1, device=images.device)
        affected = (affected < 0.5).float()
    else:
        affected = torch.ones(images.size(0), 1, 1, 1, device=images.device)

    target_labels = labels
    target_encourage = False
    if opts.encourage_labels:
        if isinstance(target_labels, list):
            target_labels = torch.LongTensor(target_labels).to(images.device)
        target_encourage = True
        assert not opts.use_l2min, 'Cannot combine l2_min with encourage_labels'

    size = opts.eps * opts.eps_overshoot
    mom = images.new_zeros(images.size())

    track_best = True if opts.use_l2min else False
    extra_steps = 1 if track_best else 0

    if opts.ensure_proper_minimization:
        first_ok_steps = images.new_zeros(images.size(0),
                dtype=torch.int).fill_(999 + opts.steps)

    if track_best:
        # Also track the first step at which results were OK, and the
        # lowest-perturbation delta.
        best_ok_deltas = torch.zeros_like(images)
        best_ok_deltas_sqr = images.new_zeros(images.size(0)).fill_(1e30)

    for step in range(opts.steps + extra_steps):
        guesses = m(images + deltas)
        loss = torch.nn.functional.cross_entropy(guesses, target_labels,
                reduction='none')
        # detach() not necessary, but just in case it changes...
        image_grads = torch.autograd.grad(loss.sum(), deltas)[0].detach()

        if isinstance(opts.use_l2min, str):
            if opts.use_l2min == 'btr':
                sm = guesses.softmax(1)
                follow_loss = (sm.gather(1, target_labels.unsqueeze(1))[:, 0]
                        > 1/sm.size(1)).float()
            else:
                raise NotImplementedError(opts.use_l2min)
        elif opts.use_l2min:
            # l2_min; aim to be correct
            follow_loss = (guesses.argmax(1) == target_labels).float()
        else:
            # Standard l2; aim to be within eps.
            follow_loss = (deltas.detach().pow(2).mean((1, 2, 3))
                    < opts.eps ** 2).float()

        if opts.ensure_proper_minimization:
            # Note that follow_loss == 0 checks for the boundary condition
            # being satisfied.
            is_ok = (follow_loss == 0)
            set_step = is_ok * (first_ok_steps > step)
            first_ok_steps[set_step] = step

        if track_best:
            # Track the best perturbation which satisfies the criteria.
            ddist = deltas.detach().pow(2).mean((1, 2, 3))
            set_best = (follow_loss == 0) * (ddist < best_ok_deltas_sqr)
            best_ok_deltas[set_best] = deltas.detach()[set_best]
            best_ok_deltas_sqr[set_best] = ddist[set_best]

            # All done; are we in extra step?  If so, don't do another step
            if step == opts.steps:
                deltas = best_ok_deltas
                break

        if target_encourage:
            image_grads *= -1

        # Let the GC clean up, if needed
        guesses = None
        loss = None

        # Ramp step size
        ss = size * (opts.steps - step) / (opts.steps * (opts.steps + 1) / 2)
        # Find direction and follow it an amount that increases RMSE up to ss
        follow_loss = follow_loss.view(-1, 1, 1, 1)
        sdir = (
                follow_loss * image_grads
                + (1 - follow_loss) * -deltas.detach())
        sdir /= 1e-8 + sdir.pow(2).mean((1, 2, 3)).sqrt().view(-1, 1, 1, 1)
        # Momentum helps when steps are high (450), hurts when steps are
        # low (45).  Difference is not necessarily significant, however.
        if opts.momentum > 0:
            mom = mom.mul_(opts.momentum).add_(sdir, alpha=1 - opts.momentum)
            sdir = mom.clone()
        sdir *= affected
        deltas.data.add_(sdir, alpha=ss)

    if opts.ensure_proper_minimization:
        if (first_ok_steps > opts.steps // 2).sum().item() != 0:
            raise ValueError("In order to ensure that a good bound is "
                    "located, calculating ARA or generating an explanation "
                    "checks that a suitable adversarial example is found "
                    "within half the specified number of optimization steps.  "
                    "This ensures that the reported minimal attack or "
                    "explanation is sufficiently representative.  "
                    "However, this was not the case.  Increase eps or steps.")

    return images + deltas.detach()


def _get_dataset():
    """Returns (ds_train, ds_test) with augmentations on training set.
    """
    T = torchvision.transforms
    aug_pad = 4
    aug_dim = 32 + aug_pad * 2
    ds_train = torchvision.datasets.CIFAR10(cifar10_path, train=True,
            transform=T.Compose([
                T.ToTensor(),
                T.Lambda(lambda tensor:
                    torch.nn.functional.pad(
                        tensor.view(1, 3, 32, 32),
                        (aug_pad,)*4,
                        'replicate').view(3, aug_dim, aug_dim)),
                T.ToPILImage(),
                T.RandomHorizontalFlip(),
                T.RandomCrop((32, 32)),
                T.ToTensor(),
            ]))
    ds_test = torchvision.datasets.CIFAR10(cifar10_path, train=False,
            transform=torchvision.transforms.ToTensor())
    return ds_train, ds_test


def _model_evaluate(m, ds_loader):
    device = next(m.parameters()).device
    m.eval()
    stats = {'accuracy': 0.}
    n = 0
    for batch in ds_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        n += images.size(0)

        preds = m(images)
        preds = preds.argmax(1)
        stats['accuracy'] += (preds == labels).float().sum().item()

        if ONE_BATCH_ONLY:
            break
    stats['accuracy'] /= n
    return stats


def _model_robust_get_grads(guesses, images, labels, train):
    """Get the gradients of guesses with respect to images, according to K=1
    and ROBUST_ZETA.

    ``train`` specifies whether or not the model is in training mode; if it is,
    the gradient tree will be retained.
    """
    to_smooth = None  # Output, or combination of outputs, for gradient

    # Select randomly smoothed nodes
    smg = torch.rand(guesses.size(0), guesses.size(1),
            device=guesses.device)
    smg = smg.argmax(1)
    sfair = guesses.gather(1, smg.unsqueeze(1))[:, 0]

    # Determine if any should be smoothed in tandem
    if ROBUST_ZETA > 0:
        # Determine which, if any, should be unfair
        unfair = (torch.rand(guesses.size(0), device=guesses.device)
                < ROBUST_ZETA).float()

        # Select unfairly smoothed
        unfair_real = guesses.gather(1, labels.unsqueeze(1))[:, 0]
        unfair_max = guesses.clone()
        unfair_max_min = torch.empty_like(unfair_max).fill_(
                unfair_max.detach().min() - 1)
        unfair_max.scatter_(1, labels.unsqueeze(1), unfair_max_min)
        sunfair = (unfair_real - unfair_max.max(1)[0])

        to_smooth = unfair * sunfair + (1 - unfair) * sfair
    else:
        # Smooth only single outputs
        to_smooth = sfair

    # Calculate gradients of input with respect to to_smooth
    grads = torch.autograd.grad(to_smooth.sum(), images, create_graph=train)[0]
    return grads


def _model_robust_get_psi(robust_integration):
    return ROBUST_ADAPT_PSI_0 * math.exp(ROBUST_ADAPT_PSI * robust_integration)


def _model_save(path, m, training_options):
    torch.save({'model_params': m.state_dict(),
            'training_options': training_options}, path)


def _model_load(path):
    d = torch.load(path)
    m = model.Model(d['training_options'])
    m.load_state_dict(d['model_params'])
    return m


def _model_train_batch(m, training_options, opt, images, labels):
    stats = {}

    F = torch.nn.functional

    images.requires_grad_()  # Require gradients for our robustness additions
    guesses = m(images)
    class_loss = -F.log_softmax(guesses, 1).gather(1, labels.unsqueeze(1))[:, 0]
    stats['class_loss_sum'] = class_loss.sum().item()

    loss = class_loss.mean()

    if training_options['robust_additions']:
        # Apply the Lipschitz Continuity loss as defined in the paper
        ri = training_options.get('robust_integration', 0.)
        # Update summation in equation from adaptive psi section
        ri += max(-ROBUST_ADAPT_EPS_NEG, min(ROBUST_ADAPT_EPS_POS,
                -math.log(loss.item() / ROBUST_ADAPT_L_TARGET)))
        ri = max(0., ri)
        training_options['robust_integration'] = ri
        input_grads = _model_robust_get_grads(guesses, images, labels,
                train=True)
        robust_loss = _model_robust_get_psi(ri) * input_grads.abs().pow(
                ROBUST_Z).mean()
        loss = loss + robust_loss

    opt.zero_grad()
    loss.backward()
    opt.step()

    return stats


### Script conclusion.
if __name__ == '__main__':
    main()

