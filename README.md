# online-offline-CGMM-for-MVDR

Online/offline CGMM implementation of the paper:

Higuchi, Takuya, et al. "Online MVDR beamformer based on complex Gaussian mixture model with spatial prior for noise robust ASR." IEEE/ACM Transactions on Audio, Speech, and Language Processing 25.4 (2017): 780-793.

Besides the commonly used offline CGMM, we also implemented online CGMM with spatial prior mentioned in paper.

---
## Files
**cgmm.py**:
1. `class CGMM` which is used as offline mode. Using EM to do Maximal Likelihood (ML) estimation
2. `class PriorCGMM(CGMM)` which is used as online mode. Using EM to do Maximal A Posterior (MAP) estimation.

**run-offline-cgmm-mvdr.py**:
Psuedo-codes of using offline CGMM (`CGMM`) to do MVDR

**run-online-cgmm-mvdr.py**:
Psuedo-codes of using online CGMM (`PriorCGMM`) to do MVDR

---
## Usage

Offline manner: see `run-offline-cgmm-mvdr.py`

Online manner: see `run-online-cgmm-mvdr.py`
