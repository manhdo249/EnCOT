import os
import numpy as np

from fastopic import FASTopic
import logging


class FASTopicTrainer:
    def __init__(self,
                 dataset,
                 num_topics=50,
                 num_top_words=15,
                 preprocessing=None,
                 epochs=200,
                 learning_rate=0.002,
                 DT_alpha=3.0,
                 TW_alpha=2.0,
                 theta_temp=1.0,
                 verbose=False
                ):
        self.dataset = dataset
        self.num_top_words = num_top_words

        self.logger = logging.getLogger('main')
        self.model = FASTopic(num_topics=num_topics,
                              preprocessing=preprocessing,
                              num_top_words=num_top_words,
                              epochs=epochs,
                              learning_rate=learning_rate,
                              DT_alpha=DT_alpha,
                              TW_alpha=TW_alpha,
                              theta_temp=theta_temp,
                              verbose=verbose
                            )
        

        if verbose:
            self.logger.setLevel("DEBUG")
        else:
            self.logger.setLevel("WARNING")

    def train(self):
        return self.model.fit_transform(self.dataset.train_texts)

    def test(self, texts):
        theta = self.model.transform(texts)
        return theta

    def get_beta(self):
        beta = self.model.get_beta()
        return beta
    
    def save_beta(self, dir_path):
        beta = self.get_beta()
        np.save(os.path.join(dir_path, "beta.npy"), beta)
        return beta

    def get_top_words(self, num_top_words=None):
        if num_top_words is None:
            num_top_words = self.num_top_words
        return self.model.get_top_words(num_top_words)

    def save_top_words(self, dir_path, num_top_words=None):
        if num_top_words is None:
            num_top_words = self.num_top_words

        top_words = self.get_top_words(num_top_words)

        top_words_path =  os.path.join(dir_path, f'top_words_{num_top_words}.txt')
        with open(top_words_path, 'w') as f:
            for _, words in enumerate(top_words):
                f.write(words + '\n')
        
        return top_words, top_words_path

    def export_theta(self):
        train_theta = self.test(self.dataset.train_texts)
        test_theta = self.test(self.dataset.test_texts)
        return train_theta, test_theta
    
    def save_theta(self, dir_path):
        train_theta, test_theta = self.export_theta()
        np.save(os.path.join(dir_path, "train_theta.npy"), train_theta)
        np.save(os.path.join(dir_path, "test_theta.npy"), test_theta)

        train_argmax_theta = np.argmax(train_theta, axis=1)
        test_argmax_theta = np.argmax(test_theta, axis=1)
        np.save(os.path.join(dir_path, 'train_argmax_theta.npy'), train_argmax_theta)
        np.save(os.path.join(dir_path, 'test_argmax_theta.npy'), test_argmax_theta)
        return train_theta, test_theta     