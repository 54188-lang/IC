import os
import shutil
from pathlib import Path
import torch
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
import random
import audio_utils
import conv_tasnet_ic
import data_manager
from config import config
from utils import print_to_file
from utility import sdr

# ---------- 工具：询问日志目录 ----------
def ask_logdir(logdir: Path):
    """
    返回 (start_epoch, ckpt_path)
    y: 删除目录并从头开始
    n: 退出
    c: 继续训练
    """
    events = list(logdir.glob('events.out.tfevents.*'))
    ckpts  = list(logdir.glob('*.pt'))

    # 自动寻找最大 epoch 的 checkpoint
    latest_epoch = 0
    latest_ckpt  = None
    for f in ckpts:
        try:
            ep = int(f.stem.split('_')[-1])  # 形如 max_20.pt
            if ep > latest_epoch:
                latest_epoch = ep
                latest_ckpt  = f
        except ValueError:
            if f.name == 'max.pt':
                latest_ckpt = f
                latest_epoch = 0

    while True:
        s = input(
            f'"{logdir}" 已有文件。\n'
            'y=删除并从头开始训练, n=退出, c=继续训练: '
        ).strip().lower()

        if s == 'y':
            shutil.rmtree(logdir, ignore_errors=True)
            logdir.mkdir(exist_ok=True)
            # 重新生成最新 config.txt，覆盖旧文件
            config_file = logdir / 'config.txt'
            print_to_file(config_file, config.print_params)
            # loss.txt 清空（或保留，这里选择清空）
            (logdir / 'loss.txt').write_text('')  # 清空旧记录
            return 0, None

        elif s == 'n':
            exit()

        elif s == 'c':
            if latest_ckpt is None:
                print("未找到 .pt 文件，继续训练从头开始。")
                return 0, None
            else:
                print(f"继续训练，起始 epoch={latest_epoch}, 加载 {latest_ckpt}")
                # 不删除任何文件，config.txt 重新生成
                config_file = logdir / 'config.txt'
                print_to_file(config_file, config.print_params)
                # loss.txt 追加模式，无需改动
                return latest_epoch, latest_ckpt

        else:
            print("请输入 y / n / c")


# ---------- Runner ----------
class Runner(object):
    def __init__(self, config):
        print("Initializing model...")
        self.model = conv_tasnet_ic.TasNet()
        total_params = sum(p.numel() for p in self.model.parameters())
        total_mb = total_params * 4 / (1024 ** 2)
        # print(f'\tTotal param #: {total_params}')
        print(f'\tTotal param size: {total_mb:.2f} MB')
        print(f'\tBatch size: {config.batch_size}')
        print(f'\tlogdir: {config.logdir}')

        self.writer = SummaryWriter(logdir=config.logdir)
        self.criterion = sdr.negative_SI_SDR()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.writer.add_text('Text', 'Parameter #: ' + str(total_params), 0)
        self.writer.add_text('Text', 'Batch size: ' + str(config.batch_size), 0)

        self._init_device(config.device, config.out_device)

        save_config_path = Path(self.writer.logdir, 'config.txt')
        if not save_config_path.exists():
            print_to_file(save_config_path, config.print_params)

    # ---------- 设备 ----------
    def _init_device(self, device, out_device):
        if device == 'cpu':
            self.in_device = torch.device('cpu')
            self.out_device = torch.device('cpu')
            self.str_device = 'cpu'
            return

        # device type: List[int]
        if type(device) == int:
            device = [device]
        elif type(device) == str:
            device = [int(device[-1])]
        else:  # sequence of devices
            if type(device[0]) != int:
                device = [int(d[-1]) for d in device]

        self.in_device = torch.device(f'cuda:{device[0]}')

        if len(device) > 1:
            if type(out_device) == int:
                self.out_device = torch.device(f'cuda:{out_device}')
            else:
                self.out_device = torch.device(out_device)
            self.str_device = ', '.join([f'cuda:{d}' for d in device])

            self.model = torch.nn.DataParallel(self.model,
                                               device_ids=device,
                                               output_device=self.out_device)
        else:
            self.out_device = self.in_device
            self.str_device = str(self.in_device)

        self.model.cuda(self.in_device)
        # 损失函数移动到设备（如果需要）
        self.criterion = self.criterion.to(self.out_device)

        torch.cuda.set_device(self.in_device)

    # ---------- 主循环 ----------
    def run(self, dataloader, stage: str, epoch: int):
        self.model.train() if stage == 'train' else self.model.eval()
        avg_loss = avg_eval = 0.
        pbar = tqdm(dataloader, desc=f'{stage}: {epoch:3d}', dynamic_ncols=True)
        for _, (x_, y_) in enumerate(pbar):
            x_, y_ = x_.to(self.in_device), y_.to(self.out_device)

            if stage == 'train':
                pred = self.model(x_)
            else:
                with torch.no_grad():
                    pred = self.model(x_)
                    
            loss = self.criterion(pred, y_)
            with torch.no_grad():
                eval_val = sdr.batch_si_sdr(pred, y_).mean().item()
                loss_val = loss.item()

            if stage == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_loss += loss_val * dataloader.batch_size
            avg_eval += eval_val * dataloader.batch_size
            pbar.set_description(f'{stage}: {epoch:3d} | Loss: {loss_val:.4f} | SI-SDR: {eval_val:.4f}')
        avg_loss /= len(dataloader.dataset)
        avg_eval /= len(dataloader.dataset)
        print(f'[{stage} Loss: {avg_loss:.4f} | SI-SDRi: {avg_eval:.4f}]')
        return avg_loss, avg_eval

# ---------- 主程序 ----------
def trainer():
    dat_num = 7
    print("Initializing data loaders...")
    train_loader, valid_loader, test_loader = data_manager.get_dataloader(config)

    # 询问日志目录
    start_epoch, ckpt_path = ask_logdir(Path(config.logdir))
    runner = Runner(config)

    # 在 trainer() 里加载 checkpoint 处
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location=runner.in_device)
        # 给所有键加前缀
        checkpoint = {f"module.{k}": v for k, v in checkpoint.items()}
        runner.model.load_state_dict(checkpoint, strict=False)
        print("Loaded checkpoint:", ckpt_path)
    else:
        start_epoch = 0

    # 训练
    for epoch in range(start_epoch + 1, config.num_epochs):
        train_loss, train_eval = runner.run(train_loader, 'train', epoch)
        valid_loss, valid_eval = runner.run(valid_loader, 'valid', epoch)

        runner.writer.add_scalar('loss/train', train_loss, epoch)
        runner.writer.add_scalar('eval/train', train_eval, epoch)
        runner.writer.add_scalar('loss/valid', valid_loss, epoch)
        runner.writer.add_scalar('eval/valid', valid_eval, epoch)

        audio_utils.train_writing(runner, train_loader, dat_num, 'train', epoch, runner.writer)
        audio_utils.train_writing(runner, valid_loader, dat_num, 'valid', epoch, runner.writer)

        # 保存 best
        if epoch == 0 or valid_loss < getattr(trainer, 'min_valid_loss', float('inf')):
            trainer.min_valid_loss = valid_loss
            save_path = Path(runner.writer.logdir, f'max_{epoch}.pt')
            if isinstance(runner.model, torch.nn.DataParallel):
                torch.save(runner.model.module.state_dict(), save_path)
            else:
                torch.save(runner.model.state_dict(), save_path)
            print(f"Epoch {epoch} - Best model saved ({save_path})")

        with open(Path(runner.writer.logdir, 'loss.txt'), 'a') as f:
            f.write(f"{epoch}: train_loss:{train_loss:.4f}, valid_loss:{valid_loss:.4f}\n")

    print('Training finished!')
    runner.writer.close()


# ---------- 入口 ----------
if __name__ == '__main__':
    random_seed = 365
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    trainer()