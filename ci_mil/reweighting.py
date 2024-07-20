import loss_reweighting as loss_expect
import torch
import torch.nn as nn
from torch.autograd import Variable

from schedule import lr_setter

@torch.enable_grad() 
def weight_learner(cfeatures, pre_features, pre_weight1, args, global_epoch=0, iter=0):
    softmax = nn.Softmax(0)
    weight = Variable(torch.ones(cfeatures.size()[0], 1).cuda(args.gpu))
    weight.requires_grad = True
    cfeaturec = Variable(torch.FloatTensor(cfeatures.size()).cuda(args.gpu))
    cfeaturec.data.copy_(cfeatures.data)
    all_feature = torch.cat([cfeaturec, pre_features[:cfeatures.size()[0]].detach()], dim=0)
    optimizerbl = torch.optim.SGD([weight], lr=args.lrbl, momentum=0.9)

    for epoch in range(args.epochb):
        lr_setter(optimizerbl, epoch, args, bl=True)
        all_weight = torch.cat((weight, pre_weight1[:cfeatures.size()[0]].detach()), dim=0)
        optimizerbl.zero_grad()

        lossb = loss_expect.lossb_expect(all_feature, softmax(all_weight), args.num_f, args.sum, args.gpu)
        print('dec_train_loss:', epoch, lossb.item())
        lossp = softmax(weight).pow(args.decay_pow).sum()
        lambdap = args.lambdap * max((args.lambda_decay_rate ** (global_epoch // args.lambda_decay_epoch)),
                                     args.min_lambda_times)
        lossg = lossb / lambdap + lossp
        if global_epoch == 0:
            lossg = lossg * args.first_step_cons

        lossg.backward(retain_graph=True)
        optimizerbl.step()

    if global_epoch == 0 and iter < 10:
        pre_features[:cfeatures.size()[0]] = (pre_features[:cfeatures.size()[0]] * iter + cfeatures) / (iter + 1)
        pre_weight1[:cfeatures.size()[0]] = (pre_weight1[:cfeatures.size()[0]] * iter + weight) / (iter + 1)

    elif cfeatures.size()[0] < pre_features.size()[0]:
        pre_features[:cfeatures.size()[0]] = pre_features[:cfeatures.size()[0]] * args.presave_ratio + cfeatures * (
                    1 - args.presave_ratio)
        pre_weight1[:cfeatures.size()[0]] = pre_weight1[:cfeatures.size()[0]] * args.presave_ratio + weight * (
                    1 - args.presave_ratio)

    else:
        pre_features = pre_features * args.presave_ratio + cfeatures * (1 - args.presave_ratio)
        pre_weight1 = pre_weight1 * args.presave_ratio + weight * (1 - args.presave_ratio)

    #softmax_weight = softmax(weight)

    return weight, pre_features, pre_weight1



@torch.enable_grad() 
def weight_learner2(cfeatures, pre_features, pre_weight1, args, global_epoch=0, iter=0):
    softmax = nn.Softmax(0)
    weight = Variable(torch.ones(cfeatures.size()[0], 1).cuda(args.gpu))
    weight.requires_grad = True
    cfeaturec = Variable(torch.FloatTensor(cfeatures.size()).cuda(args.gpu))
    cfeaturec.data.copy_(cfeatures.data)
    all_feature = torch.cat([cfeaturec, pre_features[:cfeatures.size()[0]].detach()], dim=0)
    optimizerbl = torch.optim.SGD([weight], lr=args.lrbl, momentum=0.9)

    for epoch in range(args.epochb):
        lr_setter(optimizerbl, epoch, args, bl=True)
        all_weight = torch.cat((weight, pre_weight1[:cfeatures.size()[0]].detach()), dim=0)
        optimizerbl.zero_grad()

        lossb = loss_expect.lossb_expect(all_feature, softmax(all_weight), args.num_f, args.sum, args.gpu)
        lossp = softmax(weight).pow(args.decay_pow).sum()
        lambdap = args.lambdap * max((args.lambda_decay_rate ** (global_epoch // args.lambda_decay_epoch)),
                                     args.min_lambda_times)
        lossg = lossb / lambdap + lossp
        if global_epoch == 0:
            lossg = lossg * args.first_step_cons

        lossg.backward(retain_graph=True)
        optimizerbl.step()

    if global_epoch == 0 and iter < 10:
        pre_features[:cfeatures.size()[0]] = (pre_features[:cfeatures.size()[0]] * iter + cfeatures) / (iter + 1)
        pre_weight1[:cfeatures.size()[0]] = (pre_weight1[:cfeatures.size()[0]] * iter + weight) / (iter + 1)

    elif cfeatures.size()[0] < pre_features.size()[0]:
        pre_features[:cfeatures.size()[0]] = pre_features[:cfeatures.size()[0]] * args.presave_ratio + cfeatures * (
                    1 - args.presave_ratio)
        pre_weight1[:cfeatures.size()[0]] = pre_weight1[:cfeatures.size()[0]] * args.presave_ratio + weight * (
                    1 - args.presave_ratio)

    else:
        pre_features = pre_features * args.presave_ratio + cfeatures * (1 - args.presave_ratio)
        pre_weight1 = pre_weight1 * args.presave_ratio + weight * (1 - args.presave_ratio)

    #softmax_weight = softmax(weight)

    return weight