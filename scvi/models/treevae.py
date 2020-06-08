# -*- coding: utf-8 -*-
"""Main module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as kl

from scvi.models.log_likelihood import log_zinb_positive, log_nb_positive
from scvi.models.modules import Encoder, DecoderSCVI, LinearDecoderSCVI
from scvi.models.utils import one_hot
from scvi.models.vae import VAE

import numpy as np

torch.backends.cudnn.benchmark = True

from ete3 import Tree

# TreeVAE Model
class TreeVAE(VAE):
    r"""Model class for fitting a VAE to scRNA-seq data with a tree prior. 

    This is corresponding VAE class for our TreeTrainer & implements the TreeVAE model. This model
    performs training in a very specific way, in an effort to respect the tree structure. Specifically,
    we'll perform training of this model by identifying 'clades' (or groups of leaves underneath a given
    internal node) from which the cell's RNA-seq data is assumed to be iid. This is currently done crudely
    by treating every internal node at depth 3 from the root as an appropriate location to create a clade,
    though this should be improved (see TODOs). 

    After creating a clustered subtree (where now the leaves correspond to the nodes where clades were induced),
    our training procedure is relativley simple. For every one of these new leaves, split the cells in this clade
    into train/test/validation and in each iteration sample a single cell from the appropriate list and assign its
    RNAseq profile to the clade's root (i.e., the leaf in the clusterd subtree). 

    There are a couple of items to clean up here: 

    TODO:
        - Find a more ideal way to cluster cells together into clades (currently, we're just
         using depth = 3 as the clustering rule)
        - Ensure that all math here is correct.
        - Implement the ability to sample from the posterior distribution (this is necessary for 
        inferring ancestral, or unobserved, transcriptomic profiles)

	"""

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        reconstruction_loss: str = "zinb",
        tree: Tree = None,
    ):

        super().__init__(
            n_input,
            n_batch,
            n_labels,
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            dispersion,
            log_variational,
            reconstruction_loss,
        )

        def cut_tree(node, distance):

            return node.distance == distance

        # Cluster tree into clades
        collapsed_tree = Tree(tree.write(is_leaf_fn=lambda x: cut_tree(x, 3)))
        for l in collapsed_tree.get_leaves():
            l.cells = tree.search_nodes(name=l.name)[0].get_leaf_names()

        self.root = collapsed_tree.name

        # add prior node
        inf_tree = Tree("prior_root;")
        inf_tree.add_child(collapsed_tree)

        self.prior_root = inf_tree.name

        self.tree = inf_tree

    def initialize_messages(self, evidence, barcodes, d):

        dic_nu = {}
        dic_mu = {}
        dic_log_z = {}

        for i, j in enumerate(evidence):
            dic_nu[barcodes[i]] = 0
            dic_log_z[barcodes[i]] = 0
            dic_mu[barcodes[i]] = j

        dic_nu[self.prior_root] = 0
        dic_mu[self.prior_root] = torch.from_numpy(np.zeros(d)).type(torch.DoubleTensor)
        dic_log_z[self.prior_root] = 0

        for n in self.tree.traverse():
            if n.name in dic_nu:
                n.add_features(
                    nu=dic_nu[n.name],
                    mu=dic_mu[n.name].type(torch.DoubleTensor),
                    log_z=dic_log_z[n.name],
                )
            else:
                n.add_features(
                    nu=0,
                    mu=torch.from_numpy(np.zeros(d)).type(torch.DoubleTensor),
                    log_z=0,
                )

    def initialize_visit(self):

        for node in self.tree.traverse():
            node.add_features(visited=False)

    def perform_message_passing(self, root_node, d, include_prior):
        # flag the node as visited

        prior_node = self.tree & self.prior_root
        root_node.visited = True

        incoming_messages = []
        incident_nodes = [c for c in root_node.children]
        if not root_node.is_root():
            incident_nodes += [root_node.up]

        # get list of neighbors that are not visited yet
        for node in incident_nodes:
            if not node.visited and (
                node != prior_node or (node == prior_node and include_prior)
            ):
                self.perform_message_passing(node, d, include_prior)
                incoming_messages.append(node)

        # collect and return
        if len(incoming_messages) == 0:
            # nothing to do. This happens on the leaves
            return None

        elif len(incoming_messages) == 1:
            # this happens when passing through the root
            k = incoming_messages[0]
            root_node.nu = k.nu + root_node.get_distance(k)
            root_node.mu = k.mu
            root_node.log_z = 0

        elif len(incoming_messages) == 2:
            # let us give them arbitrary names k and l (could be left and right)
            k = incoming_messages[0]
            l = incoming_messages[1]

            # let us compute the updates
            k_nu_inc = k.nu + root_node.get_distance(k)
            l_nu_inc = l.nu + root_node.get_distance(l)

            root_node.nu = 1.0 / (1.0 / k_nu_inc + 1.0 / l_nu_inc)
            root_node.mu = k.mu / k_nu_inc + l.mu / l_nu_inc
            root_node.mu *= root_node.nu

            lambda_ = k_nu_inc + l_nu_inc
            root_node.log_z = -0.5 * torch.sum((k.mu - l.mu) ** 2).item() / lambda_
            root_node.log_z -= d * 0.5 * np.log(2 * np.pi * lambda_)

        elif len(incoming_messages) == 3:
            # This happens when performing the posterior predictive variant of the message passing
            # let us give them arbitrary names k, l and m
            print("here")
            k = incoming_messages[0]
            l = incoming_messages[1]
            m = incoming_messages[2]

            # let us compute the updates
            k_nu_inc = k.nu + root_node.get_distance(k)
            l_nu_inc = l.nu + root_node.get_distance(l)
            m_nu_inc = m.nu + root_node.get_distance(m)

            root_node.nu = 1.0 / (1.0 / k_nu_inc + 1.0 / l_nu_inc + 1.0 / m_nu_inc)
            root_node.mu = k.mu / k_nu_inc + l.mu / l_nu_inc + m.mu / m_nu_inc
            root_node.mu *= root_node.nu

            lambda_ = k_nu_inc * l_nu_inc + k_nu_inc * m_nu_inc + l_nu_inc * m_nu_inc
            root_node.log_z = m_nu_inc * torch.sum((k.mu - l.mu) ** 2).item()
            root_node.log_z += l_nu_inc * torch.sum((k.mu - m.mu) ** 2).item()
            root_node.log_z += k_nu_inc * torch.sum((l.mu - m.mu) ** 2).item()
            root_node.log_z *= -0.5 / lambda_
            root_node.log_z -= d * 0.5 * np.log(2 * np.pi * lambda_)

        else:
            # Here there is a problem, we might have tried to compute something wierd
            raise NotImplementedError(
                "This should not happen (more than 3). Node" + str(root_node)
            )

    def aggregate_messages_into_leaves_likelihood(self, d, add_prior):
        res = 0

        root_node = self.tree & self.root

        # agg Z messages
        for node in self.tree.traverse():
            res += node.log_z

        if add_prior:
            # add prior
            nu_inc = 1 + root_node.nu
            res += -0.5 * torch.sum(
                root_node.mu ** 2
            ).item() / nu_inc - d * 0.5 * np.log(2 * np.pi * nu_inc)
        return res

    def posterior_predictive_density(self, d):

        root_node = self.tree & self.root

        self.initialize_visit()
        self.perform_message_passing(root_node, len(root_node.mu), True)
        self.aggregate_messages_into_leaves_likelihood(delattr, add_prior=True)
        for n in self.tree.traverse():

            print("node: ", n, " expr_value: ", n.mu)

    def forward(
        self, x, local_l_mean, local_l_var, batch_index=None, y=None, barcodes=None
    ):
        r""" Returns the reconstruction loss 

		:param x: tensor of values with shape (batch_size, n_input)
		:param local_l_mean: tensor of means of the prior distribution of latent variable l
		 with shape (batch_size, 1)
		:param local_l_var: tensor of variancess of the prior distribution of latent variable l
		 with shape (batch_size, 1)
		:param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
		:param y: tensor of cell-types labels with shape (batch_size, n_labels)
		:return: the reconstruction loss and the Kullback divergences
		:rtype: 2-tuple of :py:class:`torch.FloatTensor`
		"""
        # Parameters for z latent distribution
        outputs = self.inference(x, batch_index, y)
        qz_m = outputs["qz_m"]
        qz_v = outputs["qz_v"]
        ql_m = outputs["ql_m"]
        ql_v = outputs["ql_v"]
        px_rate = outputs["px_rate"]
        px_r = outputs["px_r"]
        px_dropout = outputs["px_dropout"]
        z = outputs["z"]

        # message passing likelihood
        self.initialize_visit()
        self.initialize_messages(
            z, [l.name for l in self.tree.get_leaves()], z.shape[1]
        )
        self.perform_message_passing((self.tree & self.root), z.shape[1], False)
        mp_lik = self.aggregate_messages_into_leaves_likelihood(
            z.shape[1], add_prior=True
        )

        qz = Normal(qz_m, torch.sqrt(qz_v)).log_prob(outputs["z"]).sum(dim=-1)

        reconst_loss = (
            self.get_reconstruction_loss(x, px_rate, px_r, px_dropout) - qz + mp_lik
        )
        return reconst_loss
