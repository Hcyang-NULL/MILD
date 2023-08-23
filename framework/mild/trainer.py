

from tqdm import tqdm
from framework.mild.dataset import *
from framework.mild.model import *
from framework.mild.sample_selection import *
from optim import *
from logger import *


def train(config):
    train_loader, val_loader, test_loader, num_classes = get_dataloader(Edict(dict(config.dataset, reference=config.reference)))

    metric_update_func, metric_calc_func = get_sample_selection()

    writer, save_dir = get_logger(Edict(dict(config.logger, title=config.title)), full_config=config)

    train_config = config.train

    model = get_model(Edict(dict(config.model, num_classes=num_classes, reference=config.reference)))

    train_index = 0
    val_index = 0
    test_index = 0
    for stage, epochs in enumerate(train_config.stages):
        criterion, optimizer, scheduler = get_optimizer(model, config.optim, T_max=epochs, stage=stage)

        writer.add_scalar('selection/clean_rate', train_loader.dataset.clean_rate, stage)
        writer.add_scalar('selection/recall', train_loader.dataset.recall, stage)
        if train_loader.dataset.alpha1s != -1:
            writer.add_scalar('weibull/alpha1', train_loader.dataset.alpha1s, stage)
            writer.add_scalar('weibull/alpha2', train_loader.dataset.alpha2s, stage)
            writer.add_scalar('weibull/beta1', train_loader.dataset.beta1s, stage)
            writer.add_scalar('weibull/beta2', train_loader.dataset.beta2s, stage)
            writer.add_scalar('weibull/proportion1', train_loader.dataset.proportion1s, stage)
            writer.add_scalar('weibull/proportion2', 1.0 - train_loader.dataset.proportion1s, stage)

        for epoch in range(epochs):
            model.train()
            train_tqdm = tqdm(train_loader, leave=False)
            for x_ids, x, y_noise, y_clean in train_tqdm:
                train_tqdm.desc = f'Stage {stage+1}/{len(train_config.stages)} Epoch {epoch+1}/{epochs}'

                x, y_noise, y_clean = x.cuda(), y_noise.cuda(), y_clean.cuda()
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y_noise)
                org_loss = loss.clone().detach().cpu().numpy()
                loss = loss.mean()
                loss.backward()
                optimizer.step()

                pred = output.argmax(dim=1)

                metric_value = metric_update_func(org_loss, pred, y_noise)
                train_loader.dataset.seq_update(x_ids, metric_value)

                writer.add_scalar('train/loss', loss.item(), train_index)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], train_index)
                train_index += 1

            if val_loader is not None and (epoch % train_config.val_interval == 0 or epoch == epochs - 1):
                model.eval()
                with torch.no_grad():
                    val_acc_clean = 0
                    for x_id, x, y_clean in val_loader:
                        x, y_clean = x.cuda(), y_clean.cuda()
                        output = model(x)
                        pred = output.argmax(dim=1)
                        val_acc_clean += (pred == y_clean).float().mean()
                    val_acc_clean /= len(val_loader)
                    writer.add_scalar('val/acc_clean', val_acc_clean, val_index)
                    val_index += 1
                    if epoch == epochs - 1:
                        print(f'Stage {stage+1} Epoch {epoch+1} Val Acc: {val_acc_clean:.4f} ', end='')

            if epoch % train_config.test_interval == 0 or epoch == epochs - 1:
                model.eval()
                with torch.no_grad():
                    test_acc_clean = 0
                    for x_id, x, y_clean in test_loader:
                        x, y_clean = x.cuda(), y_clean.cuda()
                        output = model(x)
                        pred = output.argmax(dim=1)
                        test_acc_clean += (pred == y_clean).float().mean()
                    test_acc_clean /= len(test_loader)
                    writer.add_scalar('test/acc_clean', test_acc_clean, test_index)
                    test_index += 1
                    if epoch == epochs - 1:
                        print(f'Test Acc: {test_acc_clean:.4f}')
                        writer.add_scalar('test/final_acc', test_acc_clean, stage+1)

            if (epoch+1) % train_config.save_interval == 0:
                torch.save(model.state_dict(), os.path.join(save_dir, 'model_{}_{}.pth'.format(stage, epoch)))

            scheduler.step()

        train_loader, metric_threshold = train_loader.dataset.mild_selection(metric_calc_func)
        if metric_threshold is not None:
            writer.add_scalar('selection/metric-threshold', metric_threshold, stage + 1)
