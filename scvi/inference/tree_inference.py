import logging
import sys
import time

from abc import abstractmethod
from collections import defaultdict, OrderedDict
from itertools import cycle

import numpy as np
import torch

import copy
import matplotlib.pyplot as plt


from sklearn.model_selection._split import _validate_shuffle_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from tqdm import trange

from scvi.dataset import TreeDataset
from scvi.inference import Trainer
from scvi.inference.posterior import Posterior
from scvi.models import TreeVAE

logger = logging.getLogger(__name__)

plt.switch_backend("agg")


class SequentialCladeSampler(SubsetRandomSampler):
    """ A sampler that is used to feed observations to the VAE for model fitting.

    A `SequentiaCladeSampler` instance is instantiated with a subtree, which has had leaves
    collapsed to form 'clades', which are groups of observations (leaves) that we assume are 
    drawn iid. A single iteration using the SequentailCladeSampler instance will randomly sample 
    a single observation from each clade, which we will use as a batch for training our VAE.
    
    :param data_source: A list of 'clades', each of which corresponding to a 'leaf' of the model's tree.
    :param args: a set of arguments to be passed into ``SubsetRandomSampler``
    :param kwargs: Keyword arguments to be passed into ``SubsetRandomSampler``
    """

    def __init__(self, data_source, *args, **kwargs):
        super().__init__(data_source, *args, **kwargs)
        self.clades = data_source

    def __iter__(self):
        # randomly draw a cell from each clade (i.e. bunch of leaves)
        return iter([np.random.choice(l) for l in self.clades if len(l) > 0])


class TreePosterior(Posterior):
    """The functional data unit for treeVAE.

    A `TreePosterior` instance is instantiated with a model and
    a `gene_dataset`, and as well as additional arguments that for Pytorch's `DataLoader`. 
    A subset of indices can be specified, for purposes such as splitting the data into 
    train/test/validation. Each trainer instance of the `TotalTrainer` class can therefore 
    have multiple `TreePosterior` instances to train a model. A `TreePosterior` instance 
    also comes with many methods or utilities for its corresponding data.


    :param model: A model instance from class ``treeVAE``
    :param gene_dataset: A gene_dataset instance from class ``TreeDataset``
    :param clades: A list of clades (groups of cells, assumed to be iid) that we draw observations
    from while training.
    :param use_cuda: Default: ``True``
    :param data_loader_kwargs: Keyword arguments to passed into the `DataLoader`

    Examples:

    Let us instantiate a `trainer`, with a gene_dataset and a model

        >>> tree_dataset = TreeDataset(GeneExpressionDataset, tree)
        >>> treevae= treeVAE(tree_dataset.nb_genes, tree = tree_dataset.tre
        ... n_batch=tree_dataset.n_batches * use_batches, use_cuda=True)
        >>> trainer = TreeTrainer(treevae, tree_dataset)
        >>> trainer.train(n_epochs=400)
    """

    def __init__(
        self,
        model: TreeVAE,
        gene_dataset: TreeDataset,
        clades: list,
        use_cuda: bool = False,
        data_loader_kwargs: dict = dict(),
    ):
        super().__init__(
            model=model,
            gene_dataset=gene_dataset,
            data_loader_kwargs=data_loader_kwargs,
        )

        self.clades = clades
        self.barcodes = gene_dataset.barcodes

        sampler = SequentialCladeSampler(self.clades)
        batch_size = len(self.clades)
        self.data_loader_kwargs.update({"sampler": sampler, "batch_size": batch_size})

        self.data_loader = DataLoader(gene_dataset, **self.data_loader_kwargs)

    def elbo(self) -> float:
        elbo = self.compute_elbo(self.model)
        logger.debug("ELBO : %.4f" % elbo)
        return elbo

    def compute_elbo(self, vae, **kwargs):

        """The ELBO is the reconstruction error + the likelihood of the 
        Message Passing procedure on the tree. It differs from the marginal log likelihood.
		Specifically, it is a lower bound on the marginal log likelihood
		plus a term that is constant with respect to the variational distribution.
		It still gives good insights on the modeling of the data, and is fast to compute.
		"""
        # Iterate once over the posterior and compute the elbo

        elbo = 0
        print("computing elbo")
        for i_batch, tensors in enumerate(self):
            sample_batch, local_l_mean, local_l_var, batch_index, labels = tensors[:5]
            reconst_loss = vae(
                sample_batch,
                local_l_mean,
                local_l_var,
                batch_index=batch_index,
                barcodes=self.barcodes,
                y=labels,
                **kwargs
            )
            elbo += torch.sum(reconst_loss).item()
        n_samples = len(self.indices)
        return elbo / n_samples


class TreeTrainer(Trainer):

    r"""The VariationalInference class for the unsupervised training of an autoencoder
    with a latent tree structure.

	Args:
		:model: A model instance from class ``TreeVAE``
		:gene_dataset: A TreeDataset
		:train_size: The train size, either a float between 0 and 1 or an integer for the number of training samples
		 to use Default: ``0.8``.
		:test_size: The test size, either a float between 0 and 1 or an integer for the number of training samples
		 to use Default: ``None``, which is equivalent to data not in the train set. If ``train_size`` and ``test_size``
		 do not add to 1 or the length of the dataset then the remaining samples are added to a ``validation_set``.
		:n_epochs_kl_warmup: Number of epochs for linear warmup of KL(q(z|x)||p(z)) term. After `n_epochs_kl_warmup`,
			the training objective is the ELBO. This might be used to prevent inactivity of latent units, and/or to
			improve clustering of latent space, as a long warmup turns the model into something more of an autoencoder.
		:\*\*kwargs: Other keywords arguments from the general Trainer class.

	Examples:
		>>> tree_dataset = TreeDataset(GeneExpressionDataset, tree)
        >>> treevae= treeVAE(tree_dataset.nb_genes, tree = tree_dataset.tre
        ... n_batch=tree_dataset.n_batches * use_batches, use_cuda=True)
        >>> trainer = TreeTrainer(treevae, tree_dataset)
        >>> trainer.train(n_epochs=400)
	"""
    default_metrics_to_monitor = ["elbo"]

    def __init__(
        self,
        model,
        gene_dataset,
        train_size=0.8,
        test_size=None,
        n_epochs_kl_warmup=400,
        **kwargs
    ):
        super().__init__(model, gene_dataset, **kwargs)
        self.n_epochs_kl_warmup = n_epochs_kl_warmup

        self.train_set, self.test_set, self.validation_set = self.train_test_validation(
            model, gene_dataset, train_size, test_size
        )
        self.train_set.to_monitor = ["elbo"]
        self.test_set.to_monitor = ["elbo"]
        self.validation_set.to_monitor = ["elbo"]

        self.barcodes = gene_dataset.barcodes

        # initialize messages
        # n_latent = self.model.n_latent
        # null_latent = np.stack([np.array([0]*n_latent) for a in range(len(gene_dataset.barcodes))], axis=0)
        # self.model.initialize_messages(null_latent, gene_dataset.barcodes, n_latent)

    @property
    def posteriors_loop(self):
        return ["train_set"]

    def loss(self, tensors):
        """ Computes the loss of the model after a specific iteration.

        Computes the mean reconstruction loss, which is derived after a forward pass
        of the model.

        :param tensors: Observations to be passed through model
        
        :return: Mean reconstruction loss.
        """

        sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors
        reconst_loss = self.model(
            x=sample_batch,
            local_l_mean=local_l_mean,
            local_l_var=local_l_var,
            batch_index=batch_index,
            barcodes=self.barcodes,
        )
        loss = torch.mean(reconst_loss)
        return loss

    def on_epoch_begin(self):
        if self.n_epochs_kl_warmup is not None:
            self.kl_weight = min(1, self.epoch / self.n_epochs_kl_warmup)
        else:
            self.kl_weight = 1.0

    def train_test_validation(
        self,
        model: TreeVAE = None,
        gene_dataset: TreeDataset = None,
        train_size: float = 0.9,
        test_size: int = None,
        type_class=TreePosterior,
    ):
        """Creates posteriors ``train_set``, ``test_set``, ``validation_set``.
		If ``train_size + test_size < 1`` then ``validation_set`` is non-empty.

        This works a bit differently for a TreeTrainer - in order to respect the
        tree prior we need to draw our observations from within sets of cells related
        to one another (i.e in a clade).  One can think of this analagously to 
        identifying clusters from the hierarchical ordering described by the tree, and splitting
        each cluster into train/test/validation. 

        The procedure of actually clustering the tree into clades that contain several
        iid observations is done in the constructor functin for TreeVAE (scvi.models.treevae). 
        This procedure below will simply split the clades previously identified into 
        train/test/validation sets according to the train_size specified.
        
        :param model: A ``TreeVAE` model.
        :param gene_dataset: A ``TreeDataset`` instance.
		:param train_size: float, int, or None (default is 0.1)
		:param test_size: float, int, or None (default is None)
        :param type_class: Type of Posterior object to create (here, TreePosterior)
		"""

        def get_indices_in_dataset(_subset, _subset_indices, master_list):

            _cells = np.array(_subset)[np.array(_subset_indices)]
            filt = np.array(list(map(lambda x: x in _cells, master_list)))

            return list(np.where(filt == True)[0])

        model = self.model if model is None and hasattr(self, "model") else model
        gene_dataset = (
            self.gene_dataset
            if gene_dataset is None and hasattr(self, "model")
            else gene_dataset
        )

        # this is where we need to shuffle within the tree structure
        train_indices, test_indices, validate_indices = [], [], []
        barcodes = gene_dataset.barcodes

        # for each clade induced by an internal node at a given depth split into
        # train, test, and validation and append these indices to the master list
        for l in model.tree.get_leaves():
            c = l.cells
            indices = get_indices_in_dataset(c, list(range(len(c))), barcodes)
            l.indices = np.array(indices)

        # randomly split leaves into test, train, and validation sets
        for l in model.tree.get_leaves():
            leaf_bunch = l.indices
            n_train, n_test = _validate_shuffle_split(
                len(leaf_bunch), test_size, train_size
            )
            random_state = np.random.RandomState(seed=self.seed)
            permutation = random_state.permutation(leaf_bunch)
            test_indices.append(list(permutation[:n_test]))
            train_indices.append(list(permutation[n_test : (n_test + n_train)]))
            validate_indices.append(list(permutation[(n_test + n_train) :]))

        # some print statement to ensure test/train/validation sets created correctly
        print("train_leaves: ", train_indices)
        print("test_leaves: ", test_indices)
        print("validation leaves: ", validate_indices)
        return (
            self.create_posterior(
                model, gene_dataset, train_indices, type_class=type_class
            ),
            self.create_posterior(
                model, gene_dataset, test_indices, type_class=type_class
            ),
            self.create_posterior(
                model, gene_dataset, validate_indices, type_class=type_class
            ),
        )

    def create_posterior(
        self,
        model=None,
        gene_dataset=None,
        clades=None,
        indices=None,
        type_class=TreePosterior,
    ):
        """Create a TreePosterior instance for a given set of leaves.

        This is a custom TreePoserior constructor that will take in a set of leaves (i.e. a clade)
        and return a Posterior object that can be used for training.

        :param model: A ``TreeVAE` model.
        :param gene_dataset: A ``TreeDataset`` dataset that has both gene expression data and a tree.
        :param clades: A list of clades that contain sets of leaves assumed to be iid.
        :param use_cuda: Default=True.
        :param type_class: Which constructor to use (here, TreePosterior).

        :return: A ``TreePosterior`` to use for training. 
        """
        model = self.model if model is None and hasattr(self, "model") else model
        gene_dataset = (
            self.gene_dataset
            if gene_dataset is None and hasattr(self, "model")
            else gene_dataset
        )
        return type_class(
            model,
            gene_dataset,
            clades,
            use_cuda=self.use_cuda,
            data_loader_kwargs=self.data_loader_kwargs,
        )

    def train(self, n_epochs=20, lr=1e-3, eps=0.01, params=None):
        super().train(n_epochs=n_epochs, lr=lr, eps=eps, params=params)
