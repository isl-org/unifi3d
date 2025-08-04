import math


def get_lambda_lr_poly(max_epoch, lr_power, min_lr=1e-5, **kwargs):
    def lambda_lr_poly(epoch):
        assert epoch >= 0, "Epochs are negative."
        assert max_epoch > 0, "Division by zero!"
        # assert (epoch / max_epoch) < 1, "Epochs are greater than max_epoch."
        if (epoch / max_epoch) > 1:
            epoch = max_epoch - 1
        lr = (1 - epoch / max_epoch) ** lr_power if epoch < max_epoch else min_lr
        return lr

    return lambda_lr_poly


def get_lambda_lr_exp(lr_initial, gamma, **kwargs):
    def lambda_lr_exp(epoch):
        return lr_initial * (gamma**epoch)

    return lambda_lr_exp


def get_lambda_lr_fixed(**kwargs):
    def lambda_lr_fixed(epoch):
        return 1.0

    return lambda_lr_fixed


def get_lambda_warmup_decay(warmup_epochs=80, min_lr=1e-6, max_lr=1e-4, max_epoch=900):
    def lambda_warmup_decay(epoch):
        """
        Decay the learning rate with half-cycle cosine after warmup
        https://github.com/1zb/3DShape2VecSet/blob/bedbd1091664be8e2409d580c4e1630df1a37c89/util/lr_sched.py#L10
        IMPORTANT: This function only applies a multiplicative factor to the intitial lr! It is here assumed that the
                   initial lr is set to max_lr.
        """
        if epoch < warmup_epochs:
            multiplicative_factor = epoch / warmup_epochs
        else:
            multiplicative_factor = 0.5 * (
                1.0
                + math.cos(
                    math.pi * (epoch - warmup_epochs) / (max_epoch - warmup_epochs)
                )
            )

        return max(min_lr / max_lr, multiplicative_factor)

    return lambda_warmup_decay
