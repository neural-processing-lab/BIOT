import pickle
import torch
import numpy as np
import torch.nn.functional as F
import os
from scipy.signal import resample
from scipy.signal import butter, iirnotch, filtfilt
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter
from pnpl.datasets import Shafto2014, Gwilliams2022, Armeni2022

class ArmeniLoader(torch.utils.data.Dataset):
    def __init__(self, split):
            
        if split == "train":
            self.data = Armeni2022(
                data_path="/data/engs-pnpl/lina4368/armeni2022",
                preproc_path="/data/engs-pnpl/lina4368/armeni2022",
                l_freq=0.5,
                h_freq=125,
                resample_freq=250,
                notch_freq=50,
                interpolate_bad_channels=True,
                window_len=0.5,
                label="speech",
                info=["subject_id", "session", "dataset"],
                # include_subjects=["001", "003"],
                include_sessions={"001": ["001", "002"]},
                # exclude_sessions={"001": ["009", "010"], "002": ["009", "010"], "003": ["009", "010"]},
            )
        elif split == "val":
            self.data = Armeni2022(
                data_path="/data/engs-pnpl/lina4368/armeni2022",
                preproc_path="/data/engs-pnpl/lina4368/armeni2022",
                l_freq=0.5,
                h_freq=125,
                resample_freq=250,
                notch_freq=50,
                interpolate_bad_channels=True,
                window_len=0.5,
                label="speech",
                info=["subject_id", "session", "dataset"],
                include_sessions={"001": ["009"], "002": ["009"], "003": ["009"]},
            )
        elif split == "test":
            self.data = Armeni2022(
                data_path="/data/engs-pnpl/lina4368/armeni2022",
                preproc_path="/data/engs-pnpl/lina4368/armeni2022",
                l_freq=0.5,
                h_freq=125,
                resample_freq=250,
                notch_freq=50,
                interpolate_bad_channels=True,
                window_len=0.5,
                label="speech",
                info=["subject_id", "session", "dataset"],
                include_sessions={"001": ["010"], "002": ["010"], "003": ["010"]},
            )
        else:
            raise ValueError(f"Unkown split: {split}")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = torch.FloatTensor(self.data[index]["data"])
        Y = torch.IntTensor(self.data[index]["speech"])
        return X, Y

class TUABLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        # from default 200Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class CHBMITLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 256
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        # 2560 -> 2000, from 256Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class PTBLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=500):
        self.root = root
        self.files = files
        self.default_rate = 500
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, self.freq * 5, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class TUEVLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 256
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        # 256 * 5 -> 1000, from 256Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = int(sample["label"][0] - 1)
        X = torch.FloatTensor(X)
        return X, Y


class HARLoader(torch.utils.data.Dataset):
    def __init__(self, dir, list_IDs, sampling_rate=50):
        self.list_IDs = list_IDs
        self.dir = dir
        self.label_map = ["1", "2", "3", "4", "5", "6"]
        self.default_rate = 50
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        path = os.path.join(self.dir, self.list_IDs[index])
        sample = pickle.load(open(path, "rb"))
        X, y = sample["X"], self.label_map.index(sample["y"])
        if self.sampling_rate != self.default_rate:
            X = resample(X, int(2.56 * self.sampling_rate), axis=-1)
        X = X / (
            np.quantile(
                np.abs(X), q=0.95, interpolation="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        return torch.FloatTensor(X), y


class UnsupervisedPretrainLoader(torch.utils.data.Dataset):
    def __init__(self, root_prest, root_shhs):

        # prest dataset
        self.root_prest = root_prest
        exception_files = ["319431_data.npy"]
        self.prest_list = list(
            filter(
                lambda x: ("data" in x) and (x not in exception_files),
                os.listdir(self.root_prest),
            )
        )

        PREST_LENGTH = 2000
        WINDOW_SIZE = 200

        print("(prest) unlabeled data size:", len(self.prest_list) * 16)
        self.prest_idx_all = np.arange(PREST_LENGTH // WINDOW_SIZE)
        self.prest_mask_idx_N = PREST_LENGTH // WINDOW_SIZE // 3

        SHHS_LENGTH = 6000
        # shhs dataset
        self.root_shhs = root_shhs
        self.shhs_list = os.listdir(self.root_shhs)
        print("(shhs) unlabeled data size:", len(self.shhs_list))
        self.shhs_idx_all = np.arange(SHHS_LENGTH // WINDOW_SIZE)
        self.shhs_mask_idx_N = SHHS_LENGTH // WINDOW_SIZE // 5

    def __len__(self):
        return len(self.prest_list) + len(self.shhs_list)

    def prest_load(self, index):
        sample_path = self.prest_list[index]
        # (16, 16, 2000), 10s
        samples = np.load(os.path.join(self.root_prest, sample_path)).astype("float32")

        # find all zeros or all 500 signals and then remove them
        samples_max = np.max(samples, axis=(1, 2))
        samples_min = np.min(samples, axis=(1, 2))
        valid = np.where((samples_max > 0) & (samples_min < 0))[0]
        valid = np.random.choice(valid, min(8, len(valid)), replace=False)
        samples = samples[valid]

        # normalize samples (remove the amplitude)
        samples = samples / (
            np.quantile(
                np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        samples = torch.FloatTensor(samples)
        return samples, 0

    def shhs_load(self, index):
        sample_path = self.shhs_list[index]
        # (2, 3750) sampled at 125
        sample = pickle.load(open(os.path.join(self.root_shhs, sample_path), "rb"))
        # (2, 6000) resample to 200
        samples = resample(sample, 6000, axis=-1)

        # normalize samples (remove the amplitude)
        samples = samples / (
            np.quantile(
                np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        # generate samples and targets and mask_indices
        samples = torch.FloatTensor(samples)

        return samples, 1

    def __getitem__(self, index):
        if index < len(self.prest_list):
            return self.prest_load(index)
        else:
            index = index - len(self.prest_list)
            return self.shhs_load(index)

class CamCANUnsupervisedLoader(torch.utils.data.Dataset):
    def __init__(self):
        self.data = Shafto2014(
            data_path="/data/engs-pnpl/lina4368/shafto2014/cc700/meg/pipeline/release005/BIDSsep",

            # BIOT preproc
            # preproc_path="/data/engs-pnpl/lina4368/shafto2014/BIOT-preproc",
            # l_freq=0.5,
            # h_freq=100,
            # resample_freq=200,
            # notch_freq=50,
            # interpolate_bad_channels=True,
            # window_len=10,

            # O.G. preproc
            preproc_path="/data/engs-pnpl/lina4368/shafto2014/cc700/meg/pipeline/release005/BIDSsep",
            l_freq=0.5,
            h_freq=125,
            resample_freq=250,
            notch_freq=50,
            interpolate_bad_channels=True,
            window_len=0.5,
            
            info=["subject", "dataset"],
            include_subjects=[
                'CC610292', 'CC310129', 'CC720622', 'CC221755', 'CC420061', 'CC310385', 'CC610508', 'CC321428', 'CC320109', 'CC520517', 'CC510648', 'CC122620', 'CC420729', 'CC220526', 'CC410097', 'CC620262', 'CC320616', 'CC510255', 'CC620515', 'CC510115', 'CC220506', 'CC320478', 'CC712027', 'CC222185', 'CC620129', 'CC610022', 'CC221585', 'CC620592', 'CC320417', 'CC310400', 'CC610212', 'CC222956', 'CC220901', 'CC420180', 'CC721704', 'CC320759', 'CC621128', 'CC121397', 'CC420286', 'CC420356', 'CC221511', 'CC710176', 'CC520584', 'CC420261', 'CC321203', 'CC420231', 'CC223286', 'CC410091', 'CC110411', 'CC721504', 'CC710486', 'CC520980', 'CC220974', 'CC310214', 'CC221040', 'CC420094', 'CC410287', 'CC221828', 'CC320861', 'CC710679', 'CC121428', 'CC120409', 'CC510395', 'CC520083', 'CC620785', 'CC221775', 'CC720497', 'CC620549', 'CC321585', 'CC610040', 'CC710551', 'CC120319', 'CC120462', 'CC110126', 'CC510480', 'CC222264', 'CC520097', 'CC321174', 'CC721107', 'CC710214', 'CC222258', 'CC620720', 'CC420322', 'CC410169', 'CC310331', 'CC120727', 'CC710350', 'CC520552', 'CC721392', 'CC321529', 'CC320680', 'CC310410', 'CC510355', 'CC620118', 'CC621642', 'CC410354', 'CC110098', 'CC310135', 'CC220511', 'CC420566', 'CC320888', 'CC412004', 'CC510534', 'CC620314', 'CC320445', 'CC510342', 'CC710382', 'CC620279', 'CC510486', 'CC721052', 'CC510474', 'CC222797', 'CC420623', 'CC520745', 'CC710494', 'CC510323', 'CC321025', 'CC520560', 'CC121317', 'CC321594', 'CC420173', 'CC221220', 'CC110319', 'CC320361', 'CC222367', 'CC321899', 'CC723197', 'CC122172', 'CC711035', 'CC310473', 'CC320687', 'CC620152', 'CC120208', 'CC520391', 'CC223115', 'CC420454', 'CC610146', 'CC220352', 'CC110101', 'CC212153', 'CC510639', 'CC520013', 'CC610392', 'CC221209', 'CC120313', 'CC620793', 'CC520168', 'CC320893', 'CC721291', 'CC320297', 'CC710342', 'CC110056', 'CC520607', 'CC120347', 'CC621184', 'CC420244', 'CC510551', 'CC710429', 'CC510304', 'CC320428', 'CC722536', 'CC610028', 'CC410284', 'CC420197', 'CC220610', 'CC220151', 'CC410173', 'CC510256', 'CC420776', 'CC222555', 'CC320206', 'CC220635', 'CC610285', 'CC520424', 'CC221487', 'CC321976', 'CC420143', 'CC222326', 'CC510161', 'CC610594', 'CC620444', 'CC320814', 'CC310463', 'CC120470', 'CC310414', 'CC420162', 'CC620436', 'CC720516', 'CC420433', 'CC321087', 'CC610061', 'CC620164', 'CC710566', 'CC510237', 'CC120137', 'CC410286', 'CC122405', 'CC220372', 'CC520395', 'CC210182', 'CC620451', 'CC720358', 'CC121200', 'CC210657', 'CC420493', 'CC222120', 'CC520042', 'CC320574', 'CC110087', 'CC222652', 'CC120120', 'CC210422', 'CC320850', 'CC710446', 'CC720188', 'CC420241', 'CC310391', 'CC410086', 'CC620496', 'CC720103', 'CC610469', 'CC610372', 'CC220115', 'CC220323', 'CC720119', 'CC721648', 'CC510415', 'CC621118', 'CC520078', 'CC110182', 'CC120166', 'CC620026', 'CC220223', 'CC620610', 'CC210148', 'CC321595', 'CC420202', 'CC721418', 'CC620090', 'CC420217', 'CC720238', 'CC620359', 'CC620114', 'CC510433', 'CC310160', 'CC420402', 'CC220198', 'CC520673', 'CC321506', 'CC312058', 'CC120049', 'CC610099', 'CC420182', 'CC722651', 'CC420137', 'CC720290', 'CC420392', 'CC320698', 'CC620526', 'CC120264', 'CC210250', 'CC420071', 'CC221324', 'CC711245', 'CC610405', 'CC220519', 'CC621011', 'CC121411', 'CC510473', 'CC221527', 'CC720941', 'CC320661', 'CC223085', 'CC710037', 'CC320088', 'CC222496', 'CC210174', 'CC510258', 'CC610076', 'CC310052', 'CC220920', 'CC621248', 'CC321431', 'CC520775', 'CC320553', 'CC620919', 'CC520211', 'CC321291', 'CC420587', 'CC720670', 'CC320218', 'CC321073', 'CC110045', 'CC610625', 'CC221954', 'CC420348', 'CC722891', 'CC420464', 'CC520254', 'CC221336', 'CC420383', 'CC222304', 'CC721434', 'CC510163', 'CC220232', 'CC722542', 'CC720723', 'CC721377', 'CC120764', 'CC510208', 'CC220203', 'CC620572'
            ]
        )
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # (n_channels, n_samples)
        sample = self.data[index]
        sample["data"] = resample(sample["data"], 6000, axis=-1)
        return self.data[index]

def collate_fn_camcan_pretrain(batch):
    batch = list(map(lambda x: torch.FloatTensor(x["data"]), batch))
    return torch.stack(batch, 0)

def collate_fn_unsupervised_pretrain(batch):
    prest_samples, shhs_samples = [], []
    for sample, flag in batch:
        if flag == 0:
            prest_samples.append(sample)
        else:
            shhs_samples.append(sample)

    shhs_samples = torch.stack(shhs_samples, 0)
    if len(prest_samples) > 0:
        prest_samples = torch.cat(prest_samples, 0)
        return prest_samples, shhs_samples
    return 0, shhs_samples


class EEGSupervisedPretrainLoader(torch.utils.data.Dataset):
    def __init__(self, tuev_data, chb_mit_data, iiic_data, tuab_data):
        # for TUEV
        tuev_root, tuev_files = tuev_data
        self.tuev_root = tuev_root
        self.tuev_files = tuev_files
        self.tuev_size = len(self.tuev_files)

        # for CHB-MIT
        chb_mit_root, chb_mit_files = chb_mit_data
        self.chb_mit_root = chb_mit_root
        self.chb_mit_files = chb_mit_files
        self.chb_mit_size = len(self.chb_mit_files)

        # for IIIC seizure
        iiic_x, iiic_y = iiic_data
        self.iiic_x = iiic_x
        self.iiic_y = iiic_y
        self.iiic_size = len(self.iiic_x)

        # for TUAB
        tuab_root, tuab_files = tuab_data
        self.tuab_root = tuab_root
        self.tuab_files = tuab_files
        self.tuab_size = len(self.tuab_files)

    def __len__(self):
        return self.tuev_size + self.chb_mit_size + self.iiic_size + self.tuab_size

    def tuev_load(self, index):
        sample = pickle.load(
            open(os.path.join(self.tuev_root, self.tuev_files[index]), "rb")
        )
        X = sample["signal"]
        # 256 * 5 -> 1000
        X = resample(X, 1000, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = int(sample["label"][0] - 1)
        X = torch.FloatTensor(X)
        return X, Y, 0

    def chb_mit_load(self, index):
        sample = pickle.load(
            open(os.path.join(self.chb_mit_root, self.chb_mit_files[index]), "rb")
        )
        X = sample["X"]
        # 2560 -> 2000
        X = resample(X, 2000, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y, 1

    def iiic_load(self, index):
        data = self.iiic_x[index]
        samples = torch.FloatTensor(data)
        samples = samples / (
            torch.quantile(torch.abs(samples), q=0.95, dim=-1, keepdim=True) + 1e-8
        )
        y = np.argmax(self.iiic_y[index])
        return samples, y, 2

    def tuab_load(self, index):
        sample = pickle.load(
            open(os.path.join(self.tuab_root, self.tuab_files[index]), "rb")
        )
        X = sample["X"]
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y, 3

    def __getitem__(self, index):
        if index < self.tuev_size:
            return self.tuev_load(index)
        elif index < self.tuev_size + self.chb_mit_size:
            index = index - self.tuev_size
            return self.chb_mit_load(index)
        elif index < self.tuev_size + self.chb_mit_size + self.iiic_size:
            index = index - self.tuev_size - self.chb_mit_size
            return self.iiic_load(index)
        elif (
            index < self.tuev_size + self.chb_mit_size + self.iiic_size + self.tuab_size
        ):
            index = index - self.tuev_size - self.chb_mit_size - self.iiic_size
            return self.tuab_load(index)
        else:
            raise ValueError("index out of range")


def collate_fn_supervised_pretrain(batch):
    tuev_samples, tuev_labels = [], []
    iiic_samples, iiic_labels = [], []
    chb_mit_samples, chb_mit_labels = [], []
    tuab_samples, tuab_labels = [], []

    for sample, labels, idx in batch:
        if idx == 0:
            tuev_samples.append(sample)
            tuev_labels.append(labels)
        elif idx == 1:
            iiic_samples.append(sample)
            iiic_labels.append(labels)
        elif idx == 2:
            chb_mit_samples.append(sample)
            chb_mit_labels.append(labels)
        elif idx == 3:
            tuab_samples.append(sample)
            tuab_labels.append(labels)
        else:
            raise ValueError("idx out of range")

    if len(tuev_samples) > 0:
        tuev_samples = torch.stack(tuev_samples)
        tuev_labels = torch.LongTensor(tuev_labels)
    if len(iiic_samples) > 0:
        iiic_samples = torch.stack(iiic_samples)
        iiic_labels = torch.LongTensor(iiic_labels)
    if len(chb_mit_samples) > 0:
        chb_mit_samples = torch.stack(chb_mit_samples)
        chb_mit_labels = torch.LongTensor(chb_mit_labels)
    if len(tuab_samples) > 0:
        tuab_samples = torch.stack(tuab_samples)
        tuab_labels = torch.LongTensor(tuab_labels)

    return (
        (tuev_samples, tuev_labels),
        (iiic_samples, iiic_labels),
        (chb_mit_samples, chb_mit_labels),
        (tuab_samples, tuab_labels),
    )


# define focal loss on binary classification
def focal_loss(y_hat, y, alpha=0.8, gamma=0.7):
    # y_hat: (N, 1)
    # y: (N, 1)
    # alpha: float
    # gamma: float
    y_hat = y_hat.view(-1, 1)
    y = y.view(-1, 1)
    # y_hat = torch.clamp(y_hat, -75, 75)
    p = torch.sigmoid(y_hat)
    loss = -alpha * (1 - p) ** gamma * y * torch.log(p) - (1 - alpha) * p**gamma * (
        1 - y
    ) * torch.log(1 - p)
    return loss.mean()


# define binary cross entropy loss
def BCE(y_hat, y):
    # y_hat: (N, 1)
    # y: (N, 1)
    y_hat = y_hat.view(-1, 1)
    y = y.view(-1, 1)
    loss = (
        -y * y_hat
        + torch.log(1 + torch.exp(-torch.abs(y_hat)))
        + torch.max(y_hat, torch.zeros_like(y_hat))
    )
    return loss.mean()
