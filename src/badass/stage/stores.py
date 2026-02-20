from dataclasses import dataclass, field
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import ClassVar, Dict

@dataclass
class MetaComps:
	data: np.ndarray
	wave: np.ndarray
	noise: np.ndarray
	model: np.ndarray
	resid: np.ndarray


@dataclass
class StageStore:

	OUT_NAME: ClassVar[str] = 'store_result'

	ctx: object = None
	params: Dict[str,np.ndarray] = field(default_factory=dict)
	blobs: Dict[str,np.ndarray] = field(default_factory=dict)
	comps: Dict[str,np.ndarray] = field(default_factory=dict)
	meta_comps: MetaComps = None
	metrics: Dict[str,np.ndarray] = field(default_factory=dict)


	def to_dict(self):
		# The dataclass asdict function isn't great, making our own
		return {}


	def output(self):
		outfile = self.ctx.target.outdir.joinpath('results', self.OUT_NAME)
		outfile.parent.mkdir(parents=True, exist_ok=True)

		with open(outfile.with_suffix('.json'), 'w') as f:
			json.dump(self.to_dict(), f, indent=4)


		plt.style.use('default')
		fig, ax = plt.subplots()
		ax.plot(self.meta_comps.wave, self.meta_comps.data, linewidth=1.0, color='black')
		ax.plot(self.meta_comps.wave, self.meta_comps.model, linewidth=1.0, color='red')
		plt.savefig(outfile.with_suffix('.png'))



@dataclass
class TestStore(StageStore):
	pass


@dataclass
class BasinhopStore(StageStore):
	OUT_NAME: ClassVar[str] = 'basinhop_result'


@dataclass
class MCStore(StageStore):
	OUT_NAME: ClassVar[str] = 'mc_result'

	niters: int = 0
	cur_iter: int = 0
	init_done: bool = False

	params_chain: Dict[str,np.ndarray] = field(default_factory=dict)
	blobs_chain: Dict[str,np.ndarray] = field(default_factory=dict)
	metrics_chain: Dict[str,np.ndarray] = field(default_factory=dict)


	def save_iter(self):

		chain_pairs = [
			(self.params, self.params_chain),
			(self.blobs, self.blobs_chain),
			(self.metrics, self.metrics_chain),
		]

		if not self.init_done:
			for container, chain in chain_pairs:
				for item in container.keys():
					chain[item] = np.zeros(self.niters+1)

			self.init_done = True

		# TODO: better way to do this?
		for container, chain in chain_pairs:
			for item_name, item_val in container.items():
				chain[item_name][self.cur_iter] = item_val

		self.cur_iter += 1




@dataclass
class MCMCStore(StageStore):
	pass
