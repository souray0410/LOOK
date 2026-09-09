# Project independence

LOOK and Radon_Bridge are independent research applications. Shared standards do not create shared mutable scientific state. MHD_Framework stays a generic, independently versioned dependency; both applications currently pin its stable V4 release.

| Asset | Ownership |
|---|---|
| Original UKB CFP/OCT and source label exports | Shared read-only archive with accepted checksums |
| Source code, environment and framework pin | Each application separately |
| Cohort, split and label definitions | Reusable immutable specifications, with a project-local accepted manifest |
| Model initialization, pretrained/native checkpoints and fitted bases | Project-local regular-file copies, with source SHA and use records |
| Trainable weights, optimizer, RNG/BN state and checkpoints | Written only under that project's run/artifact roots |
| Logs, metrics, predictions and reports | Per project, run and attempt |
| GPU scheduling | May share a dispatcher, never a shared optimizer or implicit coupled training |

Portable registry paths are `{code_root}/{project}/{timestamp}` for source, `{data_root}/{project}/runs/{timestamp}` for results and `{data_root}/{project}/artifacts/{timestamp}` for owned references/weights. Original data stays at `{data_root}/UKBiobank/ophthalmology`. Project artifacts are not put in the source repository or on GitHub. Reused weights are copied and SHA-verified, never an alias to another project's writable checkpoint. Deleting or replacing one project's run must not break the other.

The current two-GPU validation reads the accepted historical train/development image cache, but each application loads its own copied reference weights/bases and writes its disposable checkpoint, rank logs and acceptance under its own data/project/run tree. The dispatcher contains only plans, source snapshots and job-level receipts.

Next-cycle planning stays separate: LOOK evaluates its missing-modality/correction question; R&B evaluates structured communication between independently trained native branches. Shared disease definitions, patient splits, ResNet50 building blocks and resource measurement can improve reproducibility, but losses, missingness protocols, selected parents/bases and scientific conclusions remain project-specific. Their scores are not automatically a head-to-head comparison. Expanded UKB targets and development tuning require an explicit protocol in each repository.
