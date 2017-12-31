import torch.utils.data as data
import numpy as np
from sklearn.decomposition import PCA
import os

class VideoFeatDataset(data.Dataset):
    def __init__(self, root, flist=None, frame_num=120, which_feat='both',pca=0):
        self.root = root
        self.pathlist = self.get_file_list(flist)
        self.fnum = frame_num
        self.which_feat = which_feat
        self.pca = pca

    def __getitem__(self, index):
        path  = os.path.join(self.root, self.pathlist[index])
        if self.which_feat == 'vfeat':
            vfeat = self.loader(os.path.join(path, 'vfeat.npy')).astype('float32')    #visual feature
            if self.dequantize is not None:
                vfeat = self.dequantize(vfeat)
            if self.pca > 0:
                final = np.zeros([self.pca+1, 1024])
                t = vfeat.transpose()
                v_pca = PCA(self.pca).fit_transform(t).transpose()
                final[0] = vfeat.mean(axis=0)
                final[1:self.pca+1] = v_pca
                return final
            return vfeat

        elif self.which_feat == 'afeat':
            afeat = self.loader(os.path.join(path, 'afeat.npy')).astype('float32')    #audio feature
            if self.dequantize is not None:
                afeat = self.dequantize(afeat)
            if self.pca > 0:
                final = np.zeros([self.pca+1, 128])
                t = afeat.transpose()
                a_pca = PCA(self.pca).fit_transform(t).transpose()
                final[0] = afeat.mean(axis=0)
                final[1:self.pca+1] = a_pca
                return final
            return afeat

        else:
            vfeat = self.loader(os.path.join(path, 'vfeat.npy')).astype('float32')    #visual feature
            afeat = self.loader(os.path.join(path, 'afeat.npy')).astype('float32')    #audio feature
            if self.dequantize is not None:
                vfeat = self.dequantize(vfeat)
                afeat = self.dequantize(afeat)
            if self.pca > 0:
                v_final = np.zeros([self.pca+1, 1024])
                v = vfeat.transpose()
                v_pca = PCA(self.pca).fit_transform(v).transpose()
                v_final[0] = vfeat.mean(axis=0)
                v_final[1:self.pca+1] = v_pca

                a_final = np.zeros([self.pca+1, 128])
                a = afeat.transpose()
                a_pca = PCA(self.pca).fit_transform(a).transpose()
                a_final[0] = afeat.mean(axis=0)
                a_final[1:self.pca+1] = a_pca
                return v_final, a_final
            return vfeat, afeat

    def __len__(self):
        return len(self.pathlist)

    def loader(self, filepath):
        return np.load(filepath)

    def get_file_list(self, flist):
        filelist = []
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                filepath = line.strip()
                filelist.append(filepath)
        return filelist

    def dequantize(self, feat_vector, max_quantized_value=2, min_quantized_value=-2):
        """Dequantize the feature from the byte format to the float format.
        Args:
          feat_vector: the input 1-d vector.
          max_quantized_value: the maximum of the quantized value.
          min_quantized_value: the minimum of the quantized value.
        Returns:
          A float vector which has the same shape as feat_vector.
        """
        assert max_quantized_value > min_quantized_value
        quantized_range = max_quantized_value - min_quantized_value
        scalar = quantized_range / 255.0
        bias = (quantized_range / 512.0) + min_quantized_value
        return feat_vector * scalar + bias
