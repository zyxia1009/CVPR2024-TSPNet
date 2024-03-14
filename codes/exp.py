import datetime
import json
import os
import torch
import random
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import PTAL_Dataset
from config import generate_thumos_args
from model import TSPNet_Model
from utils import PTAL_Loss, update_label
from tools import train_one_proposal_batch, test_proposal


class Exp(object):
    def __init__(self, exp_type='THUMOS14'):
        self.config = self._get_config(exp_type)
        if self.config.seed != -1:
            self._setup_seed()
        self.device = self._get_device()

    def train(self):
        train_dataset, train_loader = self._get_data(subset='train')
        test_dataset, test_loader = self._get_data(subset='test')

        model = self._get_model().to(self.device)

        criterion = self._get_criterion()
        optimizer = self._get_optimizer(model)

        loader = iter(train_loader)
        for itr in tqdm(range(1, self.config.num_itr + 1), total=self.config.num_itr):
            if (itr - 1) % (len(train_loader) // self.config.batch_size) == 0:
                loader = iter(train_loader)
            train_one_proposal_batch(model, self.device, loader, criterion, optimizer, self.config.batch_size)

            if itr % self.config.update_fre == 0:
                update_label(dataset=train_dataset, dataloader=train_loader, model=model, device=self.device, up_threshold=self.config.up_threshold)

            if itr % 100 == 0:
                test_proposal(self.config, model, self.device, test_loader, itr)

    def test(self):
        test_dataset, test_loader = self._get_data(subset='test')
        model = self._get_model().to(self.device)
        model.load_state_dict(torch.load(self.config.model_path))
        test_proposal(self.config, model, self.device, test_loader, 100)

    def _get_config(self, exp_type):
        config_dict = {
            'THUMOS14': generate_thumos_args
        }
        config = config_dict[exp_type]()

        exp_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        config.output_folder = os.path.join(config.log_path, exp_time)
        if not os.path.exists(config.output_folder):
            os.makedirs(config.output_folder)
        json.dump(vars(config), open(os.path.join(config.output_folder, 'config.json'), 'w'), indent=1)
        # save codes
        os.system(f'cp -r ./codes {config.output_folder}/codes')
        return config

    def _get_device(self):
        is_cuda = torch.cuda.is_available()
        device_id = self.config.device
        if not is_cuda or device_id == 'cpu':
            device = torch.device('cpu')
            print('device: CPU')
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = device_id
            device = torch.device(f'cuda:{device_id}')
            print(f'device: CUDA {device_id}')
        return device

    def _get_data(self, subset):
        dataset = PTAL_Dataset(
            data_path=self.config.data_path,
            subset=subset,
            modality=self.config.modality,
            num_classes=self.config.num_classes,
            feature_fps=self.config.feature_fps,
            soft_value=self.config.soft_value
        )
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=True if subset == 'train' else False,
            num_workers=self.config.num_worker
        )
        return dataset, data_loader

    def _get_model(self):
        model = TSPNet_Model(
            args=self.config
        )
        return model

    def _get_optimizer(self, model):
        optimier = optim.Adam(model.parameters(),
                              lr=self.config.lr,
                              weight_decay=self.config.weight_decay)
        return optimier

    def _get_criterion(self):
        criterion = PTAL_Loss()
        return criterion

    def _setup_seed(self):
        """
        Set random seeds for reproducibility.
        """
        random.seed(self.config.seed)
        os.environ['PYTHONHASHSEED'] = str(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
