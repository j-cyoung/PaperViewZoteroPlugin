**000**

**001**


**002**

**003**

**004**

**005**

**006**

**007**


**008**

**009**

**010**

**011**

**012**

**013**


**014**

**015**

**016**

**017**

**018**


**019**

**020**

**021**

**022**

**023**

**024**


**025**

**026**

**027**

**028**

**029**

**030**


**031**

**032**

**033**

**034**

**035**

**036**


**037**

**038**

**039**

**040**

**041**


**042**

**043**

**044**

**045**

**046**

**047**


**048**

**049**

**050**

**051**

**052**

**053**



Under review as a conference paper at ICLR 2026

# BEYOND PIXELS: EFFICIENT DATASET DISTILLATION
## VIA SPARSE GAUSSIAN REPRESENTATION


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Dataset distillation has emerged as a promising paradigm that synthesizes compact, informative datasets capable of retaining the knowledge of large-scale counterparts, thereby addressing the substantial computational and storage burdens of
modern model training. Conventional approaches typically rely on dense pixellevel representations, which introduce redundancy and are difficult to scale up.
In this work, we propose GSDD, a novel and efficient sparse representation for
dataset distillation based on 2D Gaussians. Instead of representing all pixels
equally, GSDD encodes critical discriminative information in a distilled image
using only a small number of Gaussian primitives. This sparse representation
could improve dataset diversity under the same storage budget, enhancing coverage of difficult samples and boosting distillation performance. To ensure both
efficiency and scalability, we adapt CUDA-based splatting operators for parallel
inference and training, enabling high-quality rendering with minimal computational and memory overhead. Our method is simple yet effective, broadly applicable to different distillation pipelines, and highly scalable. Experiments show
that GSDD achieves state-of-the-art performance on CIFAR-10, CIFAR-100, and
ImageNet subsets, while remaining highly efficient encoding and decoding cost.


1 INTRODUCTION


Guided by the principles of scaling laws Kaplan et al. (2020), deep learning has advanced along the
trajectory of larger models, larger datasets, and longer training, which has significantly pushed the
performance boundaries of modern models. However, this paradigm also incurs enormous demands
on computation, storage, and communication resources, creating a barrier that hinders both broad
adoption and sustainable development. Consequently, a pivotal question in the field is how to achieve
strong performance under limited computational and data resources.


Against this backdrop, dataset distillation offers a promising solution. Its goal is to distill the essential knowledge of massive datasets into a compact set of synthetic samples, enabling models
trained on these small sets to approximate the performance of those trained on the full dataset. By
adopting dataset distillation, the costs of model training, data storage, and data transmission can be
substantially reduced. Moreover, it opens up a wide range of applications, including efficient data
replay for continual learning Yang et al. (2023); Gu et al. (2024), critical data transfer in federated
learning Huang et al. (2024a), and privacy preservation Dong et al. (2022).


Dataset distillation optimizes the synthetic data itself, with the objective of minimizing the trainingaware information gap between the synthetic and original datasets. Broadly, dataset distillation can
be decomposed into two complementary modules: distillation algorithms and data parameterization. The former defines how training-aware key information, such as training gradients Zhao et al.
(2021), model parameters Cazenavette et al. (2022), or distributional statistics Zhao & Bilen (2023),
is extracted, while the latter determines the representation format of the synthetic data. Although extensive research has focused on innovating distillation algorithms, exploration of the representation
space has been relatively limited, with most approaches directly optimizing in the pixel space. Conventional methods typically adopt a na¨ıve pixel grid as the parameterization scheme. Such dense
representations fail to capture the relative importance of pixels and lead to substantial storage redundancy. Decoder-based parameterizations, in contrast, can leverage structural priors to enhance
image realism and share common information, but they inevitably restrict the optimization space


1


**054**

**055**


**056**

**057**

**058**

**059**

**060**

**061**


**062**

**063**

**064**

**065**

**066**

**067**


**068**

**069**

**070**

**071**

**072**


**073**

**074**

**075**

**076**

**077**

**078**


**079**

**080**

**081**

**082**

**083**

**084**


**085**

**086**

**087**

**088**

**089**

**090**


**091**

**092**

**093**

**094**

**095**


**096**

**097**

**098**

**099**

**100**

**101**


**102**

**103**

**104**

**105**

**106**

**107**



Under review as a conference paper at ICLR 2026


and introduce additional computational and memory overhead. More recently, approaches based
on implicit neural representations (INRs), such as DDiF Shin et al. (2025), have substantially reduced per-image storage and thereby improved the diversity of distilled datasets. However, their
inherent per-pixel query decoding mechanism incurs prohibitive computational costs, particularly in
high-resolution or large-batch scenarios, creating a severe performance bottleneck.


In this paper, we propose a new paradigm for dataset distillation, termed Gaussian Splatting Dataset
Distillation (GSDD). Instead of relying on dense pixel grids, GSDD represents each distilled image
with a sparse set of 2D Gaussians. Each Gaussian explicitly encodes training-aware image features
that span multiple pixels, and optimizes only a few parameters such as position and shape, thereby
replacing large-scale pixel-level optimization. This sparse representation significantly reduces the
storage cost per image, which in turn increases the diversity of the distilled dataset and improves
coverage across samples of varying difficulty. To further ensure the scalability of GSDD, we adapt
a highly parallelized differentiable rasterizer that enables instantaneous high-quality rendering from
parameters to images.


Our contributions can be summarized as follows:


   - We propose a novel dataset parameterization framework based on 2D Gaussian representations. This is the first work that introduces sparse Gaussian representations into dataset
distillation. The method is both simple and effective, and it can enhance the performance
of a wide range of distillation algorithms.


   - We provide a complete design of the Gaussian representation, including parameterization, optimization objectives, and an efficient batch-parallel Gaussian rasterization operator.
This design ensures fast convergence, high distillation quality, and stable optimization.


    - Extensive experiments demonstrate that our method achieves state-of-the-art performance
on CIFAR-10, CIFAR-100, and multiple ImageNet subsets, while offering faster computation and lower memory consumption compared with prior approaches.


2 BACKGROUND AND MOTIVATION


2.1 RELATED WORK


**Dataset Distillation Algorithms** Modern deep learning models typically rely on large-scale training datasets, whereas the goal of dataset distillation is to replace the dataset with a much smaller
set of high-quality training samplesWang et al. (2018). Unlike data selection, dataset distillation
explicitly optimizes the synthetic data to achieve superior performance.


Formally, let the original dataset be denoted as


_T_ = _{_ ( _ti, yi_ ) _| i_ = 1 _, · · ·, NT } ⊆D,_


where _D_ represents the underlying data distribution. The distilled dataset (also referred to as the
synthetic dataset) is defined as


_S_ = _{_ ( _sj, yj_ ) _| j_ = 1 _, · · ·, NS_ _},_


where _ti_ and _sj_ denote original and distilled images, and _yi_ and _yj_ are their corresponding labels.
Since this work adopts one-hot label encoding, without loss of generality we denote the distilled
dataset as
_S_ = _{sj | j_ = 1 _, · · ·, NS_ _}._


The storage footprint of _S_ is typically required to be far smaller than that of _T_ . In dataset distillation,
the synthetic dataset is treated as a set of learnable objects, where the pixel values of distilled images
_sj_ are regarded as optimization parameters. The ultimate goal is to train a model _θS_ on the compact
distilled dataset such that its performance closely matches that of a model _θT_ trained on the full
dataset Wang et al. (2018):


_S_ = arg min �� _l_ ( _θS_ _, D_ ) _−_ _l_ ( _θT, D_ )�� _,_ _θS_ = arg min _l_ ( _θ, S_ ) _,_ _θT_ = arg min _l_ ( _θ, T_ ) _._
_S_ _θ_ _θ_


2


**108**

**109**


**110**

**111**

**112**

**113**

**114**

**115**


**116**

**117**

**118**

**119**

**120**

**121**


**122**

**123**

**124**

**125**

**126**


**127**

**128**

**129**

**130**

**131**

**132**


**133**

**134**

**135**

**136**

**137**

**138**


**139**

**140**

**141**

**142**

**143**

**144**


**145**

**146**

**147**

**148**

**149**


**150**

**151**

**152**

**153**

**154**

**155**


**156**

**157**

**158**

**159**

**160**

**161**



Under review as a conference paper at ICLR 2026


This optimization objective can be approached via meta-learning, but such methods often consume
substantial GPU memory Feng et al. (2024) and yield suboptimal performance due to weak supervision signals. To address these challenges, many alternative distillation objectives have been
proposed. The core idea is to extract and align critical training-aware information between _S_ and
_T_ . Mainstream approaches include trajectory matching(TM) Cazenavette et al. (2022); Du et al.
(2023a); Liu et al. (2025); Guo et al. (2024); Cui et al. (2023), which constrains the distance between model parameters trained on the two datasets; distribution matching(DM) Zhao & Bilen
(2023); Zhao et al. (2023); Liu et al. (2023); Wei et al. (2024); Wang et al. (2025b), which aligns
feature distributions; and gradient matching(DC) Zhao et al. (2021); Lee et al. (2022); Du et al.
(2023b), which constrains the gradients derived from the two datasets. Despite the drastic reduction
in storage requirements, distilled datasets often achieve competitive training performance, and the
resulting synthetic images can diverge significantly from the originals. Consequently, dataset distillation has found applications in continual learning Yang et al. (2023); Gu et al. (2024), privacy
preservation Dong et al. (2022); Chung et al. (2024); Zheng et al. (2025), federated learning Huang
et al. (2024a); Jia et al. (2025); Yan et al. (2025), and neural architecture search Such et al. (2020).


**Parameterization of Dataset Distillation** The parameterization of the distilled dataset _S_ critically
influences both performance and efficiency. Early methods directly optimized RGB pixels, but this
dense representation wastes storage by ignoring spatial redundancy. To address this, decoder-based
approaches Liu et al. (2022); Liu & Wang (2023); Cazenavette et al. (2023) learn latent codes with
a shared decoder, which reduces redundancy but limits flexibility in optimization and adds computation costs. Dictionary-based methods Deng & Russakovsky (2022); Wei et al. (2023) instead
combine basis images with index matrices to support domain adaptation and generalization, though
they still store pixel-level bases and are restricted by linear combinations.


Recent studies have introduced compact low-level representations to reduce image redundancy and
increase distilled samples within a fixed storage budget, such as downscaling Kim et al. (2022),
frequency-domain transforms Shin et al. (2023), and implicit neural representations (INRs) Shin
et al. (2025). INR-based methods show strong performance by encoding each image as a neural
field with fewer parameters, but their efficiency is limited since decoding requires per-pixel RGB
queries, causing heavy overhead for high-resolution and large-batch training.


**Gaussian Splatting** Gaussian splatting Kerbl et al. (2023) represents 3D scenes using Gaussian
ellipsoids, offering higher rendering efficiency and explicit geometric control compared to NeRFbased representations Mildenhall et al. (2021). Originally developed for novel view synthesis, it
initializes a set of Gaussian primitives within a 3D scene, which are then rendered from multiple viewpoints and optimized to match the corresponding ground-truth images, ultimately enabling
arbitrary-view synthesis. In recent years, Gaussian splatting has been extended to a wide range
of domains, including digital humans Qian et al. (2024); Zielonka et al. (2025), autonomous driving Zhou et al. (2024); Huang et al. (2024b), dynamic scene modeling Wu et al. (2024); Huang et al.
(2024c), and 3D editing Chen et al. (2024). More recently, Gaussian representations have also been
applied to 2D vision tasks such as image super-resolution Hu et al. (2024); Chen et al. (2025), image
representation Zhu et al. (2025); Weiss & Bradley (2024); Zhang et al. (2025), and video representation Wang et al. (2025a), where Gaussian primitives are used to efficiently encode high-fidelity
image of videos with fine-grained textures.


This work is the first to introduce Gaussian Splatting as a parameterization for dataset distillation,
which leverages Gaussian representations as a simple yet efficient parameterization for distilled
datasets. Unlike prior approaches that rely on network structures or complex operations, GSDD
employs a CUDA-based differentiable rasterization algorithm to render sparse Gaussian primitives.
As an explicit representation, it offers high flexibility in optimization, substantially increases dataset
diversity under the same storage budget, and significantly accelerates both encoding and decoding. Rather than emphasizing fine-grained image fidelity, our method parameterizes entire distilled
datasets with Gaussian representations and directly optimizes them for model training performance.


2.2 MOTIVATION


Gaussian representations enable faster decoding and rendering, especially compared to the stateof-the-art DDiF Shin et al. (2025) method. DDiF relies on implicit neural representations using a


3


**162**

**163**


**164**

**165**

**166**

**167**

**168**

**169**


**170**

**171**

**172**

**173**

**174**

**175**


**176**

**177**

**178**

**179**

**180**


**181**

**182**

**183**

**184**

**185**

**186**


**187**

**188**

**189**

**190**

**191**

**192**


**193**

**194**

**195**

**196**

**197**

**198**


**199**

**200**

**201**

**202**

**203**


**204**

**205**

**206**

**207**

**208**

**209**


**210**

**211**

**212**

**213**

**214**

**215**



Under review as a conference paper at ICLR 2026


SIREN network Sitzmann et al. (2020) to model distilled images, where each pixel’s RGB value
must be queried individually, as illustrated in Figure 1a. This approach incurs substantial time and
memory overhead as the number and resolution of distilled images increase. In contrast, GSDD
implements a custom CUDA operator for batched rendering of Gaussians. This allows all Gaussians
to be efficiently rendered into distilled images in a single pass. As shown in Figure 1b, this design
significantly reduces both inference and optimization time and memory usage, thereby improving
the scalability of dataset distillation methods.


Gaussian representations enable compact

computation costs. In contrast, GSDD can
use a single Gaussian to represent a larger

taneously. This not only provides a more effi
optimization efficiency. As shown in Figure 1c, pixel-based methods optimize each

|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||(𝑥,|𝑦)|
|||||


Figure 1: Comparison of Different Distilled Image

pixel independently, which inevitably intro
Representation

duces redundancy and noise. In comparison, Gaussian-based optimization aggregates
pixel-level gradients within the support region of each Gaussian, resulting in more robust updates.
Furthermore, each Gaussian can perform localized optimization by adjusting a small set of parameters such as position, color, and shape. This region-level update strategy improves both efficiency
and effectiveness during training.



Thus, each Gaussian component can be fully described by a 9-dimensional vector _pk_ =
( _uk, vk, l_ 11 _[k]_ _[, l]_ 21 _[k]_ _[, l]_ 22 _[k]_ _[,]_ **[ c]** _[k][, α][k]_ [)] _[ ∈]_ [R][9] _[.]_ [ Accordingly, the parameter set of an entire distilled image] _[ s][j]_ [is]


4



pixel-level gradients











query coordinates



(a)







region-level optimization





Figure 1: Comparison of Different Distilled Image
Representation



3 METHODOLOGY


3.1 PARAMETERIZATION OF DATASET DISTILLATION BY GAUSSIAN SPLATTING


We adopt a 2D Gaussian mixture model to parameterize distilled images. Each distilled image _sj_ is
represented as a set of _M_ Gaussian components, denoted by


_Gj_ = _{gk | k_ = 1 _, · · ·, M_ _}._ (1)


Specifically, for a pixel located at coordinates ( _x, y_ ), the contribution from the _k_ -th Gaussian component _gk_ is defined by the following unnormalized Gaussian function:

_gk_ ( _x, y_ ; _µk,_ Σ _k_ ) = exp� _−_ [1] 2 [(] **[x]** _[ −]_ _[µ][k]_ [)] _[T]_ [ Σ] _k_ _[−]_ [1][(] **[x]** _[ −]_ _[µ][k]_ [)]           - _,_ (2)

where **x** = [ _x, y_ ] _[T]_ denotes the pixel center and _µk_ = [ _uk, vk_ ] _[T]_ is the Gaussian mean. The covariance
matrix Σ _k_ determines the size, shape, and orientation of the Gaussian. To ensure that Σ _k_ remains
positive semi-definite during optimization, we parameterize it via a stable Cholesky decomposition,
i.e., Σ _k_ = _LkL_ _[T]_ _k_ [, where the lower-triangular matrix] _[ L][k]_ [ is given by]



_Lk_ = - _l_ 11 _[k]_ 0
_l_ 21 _[k]_ _l_ 22 _[k]_




_._ (3)



Each Gaussian _gk_ is associated with a color vector **c** _k ∈_ R [3] and an opacity _αk_ . The final color of
pixel ( _x, y_ ) is then synthesized by aggregating the contributions of all Gaussians:



**c** ( _x, y_ ) =



_M_

- _αk · gk_ ( _x, y_ ; _uk, vk,_ Σ _k_ ) _·_ **c** _k._ (4)


_k_ =1


Under review as a conference paper at ICLR 2026



**216**

**217**


**218**

**219**

**220**

**221**

**222**

**223**


**224**

**225**

**226**

**227**

**228**

**229**


**230**

**231**

**232**

**233**

**234**


**235**

**236**

**237**

**238**

**239**

**240**


**241**

**242**

**243**

**244**

**245**

**246**


**247**

**248**

**249**

**250**

**251**

**252**


**253**

**254**

**255**

**256**

**257**


**258**

**259**

**260**

**261**

**262**

**263**


**264**

**265**

**266**

**267**

**268**

**269**


|Rasterize Images in<br>Training Stage<br>Batch<br>FP32 BF16 Pseudo FP32 …<br>Concated Par… ams Rasterize Distilled Images<br>…… … ……<br>save<br>Pseudo FP32<br>BF16 … ……<br>Inference Stage Pre-Filter SSAA<br>pixel to be colored colored pixel<br>Quant Gaussian Params Anti-alias in Distilled Images<br>forward active bytes params of one distilled images<br>backward zeroed bytes pixel with sampled points|Col2|
|---|---|
|FP32<br>**……**<br>**…**<br>**Inference Stage**<br>**save**<br>BF16<br>**……**<br>Pseudo FP32<br>**…**<br>**……**<br>**Training Stage**<br>**Pre-Filter**<br>**SSAA**<br>pixel to be colored<br>colored pixel<br>**Rasterize Images in**<br>**Batch**<br>**Quant Gaussian Params**<br>forward<br>backward<br>**…**<br>**…**<br>active bytes<br>zeroed bytes<br>params of one distilled images<br>Concated Params<br>Distilled Images<br>Pseudo FP32<br>BF16<br>**Anti-alias in Distilled Images**<br>**Rasterize**<br>pixel with sampled points||
|FP32<br>**……**<br>**…**<br>**Inference Stage**<br>**save**<br>BF16<br>**……**<br>Pseudo FP32<br>**…**<br>**……**<br>**Training Stage**<br>**Pre-Filter**<br>**SSAA**<br>pixel to be colored<br>colored pixel<br>**Rasterize Images in**<br>**Batch**<br>**Quant Gaussian Params**<br>forward<br>backward<br>**…**<br>**…**<br>active bytes<br>zeroed bytes<br>params of one distilled images<br>Concated Params<br>Distilled Images<br>Pseudo FP32<br>BF16<br>**Anti-alias in Distilled Images**<br>**Rasterize**<br>pixel with sampled points|**nti-alias in Distilled Images**|
|forward<br>backward<br>active bytes<br>zeroed bytes<br>params of one distilled images<br>pixel with sampled points|forward<br>backward<br>active bytes<br>zeroed bytes<br>params of one distilled images<br>pixel with sampled points|



Figure 2: Overview of the proposed framework. Each Gaussian is parameterized by a total of 9 floating-point
values. A single distilled image is represented by a set of Gaussians. During training, we first quantize the
parameters to bf16 precision to obtain quantized Gaussian primitives. These are then rendered into distilled
images using a customized rasterizer. During the rasterization process, we concatenate all primitives from the
distilled dataset and feed them into a single batched rasterization kernel. This design enables efficient rendering
and facilitates a compact data structure, as the entire distilled dataset can be managed by initializing only one
object instance. To further improve rendering quality when Gaussians are sparse, we incorporate prefiltering
and SuperSampling-based Anti-Aliasing techniques. These enhancements enable more accurate estimation
of RGB values at each pixel. The rendered distilled images are then aligned with the original images for
information matching, and the resulting gradients are backpropagated to update the Gaussian parameters.


given by **p** _[j]_ = _{p_ _[j]_ _k_ _[}]_ _k_ _[M]_ =1 [, and the parameterization of the whole distilled dataset] _[ S]_ [ is] _[ P][S]_ [ =] _[ {]_ **[p]** _[j][}]_ _j_ _[N]_ =1 _[S]_ _[.]_
Based on this parameterization, _R_ can be seen as a rendering function, which maps the parameters
**p** _[j]_ onto a predefined pixel coordinate grid _C_, and is implemented as the cuda operator,


_sj_ = _R_ ( **p** _[j]_ ; _C_ ) _,_ (5)


which maps the parameters **p** _[j]_ onto a predefined pixel coordinate grid _C_ . Here, _C_ denotes the set of
all pixel coordinates, i.e.,


_C_ = _{x | x ∈_ 1 _/W, · · ·,_ ( _W −_ 1) _/W_ _} × {y | y ∈_ 1 _/H, · · ·,_ ( _H −_ 1) _/H}._ (6)


This Gaussian-based parameterization is orthogonal to the choice of dataset distillation algorithm.
Once a differentiable synthesis function is defined, it can be seamlessly integrated into any distillation framework. The optimization objective is to find the optimal distilled parameters Param( _S_ ) _[∗]_
that minimize the distillation loss _L_ distill:


Param( _S_ ) _[∗]_ = arg min (7)
Param( _S_ ) _[L]_ [distill][(] _[S][,][ T]_ [ )] _[,]_


where _L_ distill can correspond to any distillation objective.


To accelerate convergence and provide a good initialization for the optimization process, we preinitialize the parameters of the distilled dataset. Specifically, we randomly sample a small subset of
real images _S_ 0 _, |S_ 0 _|_ = _NS_ from the original dataset _T_, and minimize the mean squared error (MSE)
between the synthesized images and these real samples. The initialization objective is thus:


Param( _S_ )0 = arg min (8)
Param( _S_ ) _[L]_ [MSE][(] _[S][,][ S]_ [0][)] _[.]_


3.2 EFFICIENT GAUSSIAN SPLATTING DESIGN


The core of the above optimization lies in the differentiable renderer _R_ ( **p** _[j]_ ; _C_ ), which maps Gaussian
primitives to distilled images with both high visual fidelity and computational scalability. To enable
efficient processing of Gaussian representations for large-scale dataset distillation, we incorporate


5















**Matching Loss**

𝑳𝒅𝒅


**Original Images**






























Under review as a conference paper at ICLR 2026


|75|5|Col3|Col4|Col5|70|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
|60<br>65<br>70<br>curacy (%)|||||~~  pacity~~<br>  pacity<br>  ize<br>  ize<br>20<br>30<br>40<br>50<br>60<br><br>Accuracy (%)||||
|60<br>65<br>70<br>curacy (%)|||||||||
|50<br>55<br>Ac|||~~k~~<br>k<br>k<br>k|~~eep large o~~<br>eep small o<br>eep large s<br>ep small s|~~eep large o~~<br>eep small o<br>eep large s<br>ep small s|1<br>~~10~~<br>40|~~GPC~~<br><br><br>80<br>~~16~~<br>T|~~0~~<br>|


|80<br>60 (%)<br>Accuracy<br>40<br>20|Col2|80<br>60 (%)<br>Accuracy<br>40<br>20|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|20<br>40<br>60<br>80<br>Accuracy (%)|||||||
|20<br>40<br>60<br>80<br>Accuracy (%)|GPC<br>1<br>10<br>40<br>80<br>160<br>TM|GPC<br>1<br>10<br>40<br>80<br>160<br>TM|[50, 10<br>|Difficu<br> 0]<br> <br>[<br>|lty Bin<br>30, 40)<br>|[10, 20<br>|



(d)



**270**

**271**


**272**

**273**

**274**

**275**

**276**

**277**


**278**

**279**

**280**

**281**

**282**

**283**


**284**

**285**

**286**

**287**

**288**


**289**

**290**

**291**

**292**

**293**

**294**


**295**

**296**

**297**

**298**

**299**

**300**


**301**

**302**

**303**

**304**

**305**

**306**


**307**

**308**

**309**

**310**

**311**


**312**

**313**

**314**

**315**

**316**

**317**


**318**

**319**

**320**

**321**

**322**

**323**



introducing a minimum rendering variance that suppresses aliasing for extremely narrow Gaussians;
and (ii) 2 _×_ 2 supersampling anti-aliasing (SSAA), which averages multiple samples within each
pixel area to smooth boundaries and reduce high-frequency artifacts with minimal overhead.


During optimization, some Gaussian centers tend to drift outside the normalized coordinate space

[ _−_ 1 _,_ 1] [2], where they receive no gradients and become unrecoverable, a phenomenon we term _Gaus-_
_sian Escape_ . To alleviate this, we apply a boundary regularization loss to slightly encourage Gaussian centers to remain within the viewable region:


_l_ boundary = _−_ E _k_                   - log(1 _−_ _x_ ˜ [2] ) + log(1 _−_ _y_ ˜ [2] )� _._ (10)


In practice, the positions of Gaussian primitives are defined within the range [ _−_ 1 _,_ 1], and we perform coordinate transformations inside the CUDA operator. At each iteration, we clip the Gaussian
positions to ensure they remain within the valid bounds:


_µ_ ˜ _k_ = clip( _µk, −_ 1 + _ϵ,_ 1 _−_ _ϵ_ ) _._ (11)


We observe that Gaussian parameters exhibit robustness to reduced numerical precision, as the rasterization process does not suffer from cumulative precision errors. Based on this, we store all
Gaussian parameters in bfloat16 (bf16) precision. During training, parameters are maintained in
fp32 to ensure accurate updates, while bf16 casting is applied during the forward pass to allow the
model to adapt to quantization effects. After training, all parameters are quantized to bf16.


3.3 ANALYSIS OF THE GAUSSIAN REPRESENTATION


**Sparsity improves optimization** A key advantage of using Gaussian representations for image
distillation lies in their ability to efficiently capture and encode training-aware image features that


6























(a)



(b)





(c)









Figure 3: (a) Distillation performance under different Gaussian pruning strategies as a function of the remaining Gaussian ratio; (b) Test accuracy across training epochs with different GPC (Gaussian Images Per Class)
under the same storage budget; (c) Test accuracy of the distilled dataset on samples of varying difficulty under
the same storage budget; (d) Relationship between prediction accuracy on samples of different difficulty and
GPC under the same storage budget. For fair comparison, TM is initialized with real images and serves as a
baseline that represents pixel-based distillation.


several critical designs into the rendering pipeline, including parallel rendering of multiple distilled
images, anti-aliasing strategies, spatial constraints on Gaussian positions and quantization.


Existing open-source rasterization operators are primarily designed for 3D scene reconstruction or
single-image tasks such as representation learning and super-resolution, and generally lack support for multi-image rendering. To fully leverage the parallelism of modern GPUs, we implement
customized data structures and rendering kernels for both the forward and backward passes. Specifically, all Gaussian primitives across the entire distilled dataset are represented as a single contiguous
1D vector, and a globally unique ID (GUID) system is used to assign threads that rasterize multiple
images in parallel. Details are provided in Appendix C, and code will be open-sourced.


Some Gaussians may become highly anisotropic during optimization to capture high-frequency features such as edges, which necessitates anti-aliasing strategies. We adopt two complementary techniques: (i) analytic pre-filtering, which approximates each Gaussian’s integral over a pixel area by
convolving it with a unit pixel box filter, resulting in a modified covariance



Σ _[′]_ _k_ [= Σ] _[k]_ [+ Σ] _[box][,]_ where Σ _box_ = diag - 1



12 [1] - _,_ (9)



1 [1]

12 _[,]_


**324**

**325**


**326**

**327**

**328**

**329**

**330**

**331**


**332**

**333**

**334**

**335**

**336**

**337**


**338**

**339**

**340**

**341**

**342**


**343**

**344**

**345**

**346**

**347**

**348**


**349**

**350**

**351**

**352**

**353**

**354**


**355**

**356**

**357**

**358**

**359**

**360**


**361**

**362**

**363**

**364**

**365**


**366**

**367**

**368**

**369**

**370**

**371**


**372**

**373**

**374**

**375**

**376**

**377**



Under review as a conference paper at ICLR 2026


span multiple pixels, using only a single or very few Gaussian primitives. As a result, this approach
yields a more compact and efficient representation for each distilled image. We design a pruning
experiment based on the following hypothesis: Gaussian components with larger spatial extent and
higher opacity are more likely to carry critical training information. As shown in Figure 3a, when
these large and opaque Gaussians are pruned first, the model performance drops sharply as the
pruning ratio increases. In contrast, removing smaller and more transparent Gaussians has less
effect on the performance. The results indicate that large Gaussian components not only cover
broader pixel regions but also contribute more significantly to the final performance of the distilled
dataset. This further confirms that the proposed Gaussian representation captures a highly sparse
and information-dense encoding of the distilled image.


The Gaussian representation enables efficient
modeling of distilled images with a small number of parameters. Its core mechanism lies
in the ability to apply structured and coherent
modifications to an entire pixel region by adjusting the properties of a single Gaussian component, such as its position, shape, color, and
opacity. During backpropagation, each Gaus
(a) (b)

sian also naturally aggregates the gradients of
all pixels it covers. This optimization paradigm,

Figure 4: (a) Loss landscape of GSDD; (b) Loss

which operates at the level of geometric primi
landscape of pixel-based representation.

tives, offers improved robustness and efficiency
compared to traditional pixel-wise optimization. As shown by the convergence curves in Figure 3b, GSDD achieves faster convergence and
higher final performance than pixel-based methods. To further examine the underlying differences
in optimization behavior, we visualize the loss landscapes of both approaches in Figure 4a and 4b.
GSDD exhibits a smoother loss surface with a lower minimum and a wide basin resembling that
of convex optimization, which substantially reduces the difficulty of gradient descent. In contrast,
the pixel-based method presents a highly rugged and noisy loss landscape, making the optimization
process more susceptible to poor local minima.



**Diversity improves scalability** The parameter efficiency of Gaussian representations allows more
distilled images to be synthesized under a fixed storage budget, thereby improving coverage of and
generalization to more challenging samples. Prior work has shown that models often rely on memorization to generalize to hard examples, a phenomenon that is difficult to address in conventional
dataset distillation settings. To evaluate this advantage, we conduct the following experiment: we
first construct a difficulty spectrum by counting the number of incorrect predictions made by 100
independently trained models for each test sample. We then fix the overall storage budget and compare the performance of student models trained on distilled datasets with different Gaussian images
Per Class(GPC) values, measuring their accuracy across different difficulty ranges. As shown in
Figures 3c and 3d, the results reveal a clear trend: increasing the number of representable distilled
images initially yields substantial performance gains on easy samples. With higher GPC values, the
models further improve their accuracy on harder examples. These findings suggest that GSDD enhances downstream generalization by increasing the diversity of distilled images, thereby covering
a broader range of the difficulty spectrum.


4 EXPERIMENTS


4.1 EXPERIMENTAL RESULTS


**Datasets and Baselines** We evaluate our method on standard datasets: CIFAR-10, CIFAR-100
(at 32 _×_ 32 resolution), six ImageNet subsets (at 128 _×_ 128 resolution and 256 _×_ 256 resolution),
and ImageNet-1K (at 224 _×_ 224 resolution). The general pipeline of dataset distillation involves
two stages: (1) generating a synthetic dataset, and (2) training a model from scratch on the synthetic data. The performance of the trained model on the real test set is used to assess the quality
of the distilled dataset. We include a wide range of recent dataset distillation baselines, including TM Cazenavette et al. (2022), FRePo Zhou et al. (2022), IDC Kim et al. (2022), FreD Shin
et al. (2023), HaBa Liu et al. (2022), RTP Deng & Russakovsky (2022), HMN Liu et al. (2022),


7



















(a)



(b)



Figure 4: (a) Loss landscape of GSDD; (b) Loss
landscape of pixel-based representation.


Table 2: Test accuracy on ImageNet subsets (128 _×_
128) with gradient matching (DC) and distribution
matching (DM).


**DC** ImageNet Subset (128 _×_ 128)
Avg
Nette Woof Fruit Yellow Meow Squawk


DC (Vanilla) 34.2 22.5 21.0 37.1 22.0 32.0 28.1
GLaD 35.4 22.3 20.7   - 22.6 33.8 27.0
H-GLaD 36.9 24.0 22.4  - 24.1 35.3 28.5
IDC 45.4 25.5 26.8   - 25.3 34.6 31.5
FreD 49.1 26.1 30.0   - 28.7 39.7 34.7
DDiF 61.2 35.2 37.8   - 39.1 54.3 45.5


GSDD **70.2** **42.4** **47.8** **64.0** **43.2** **69.4** **56.2**


**DM**
DM (Vanilla) 30.4 20.7 20.4 36.0 20.1 26.6 25.7
GLaD 32.2 21.2 21.8   - 22.3 27.6 25.0
H-GLaD 34.8 23.9 24.4  - 24.2 29.5 27.4
IDC 48.3 27.0 29.9   - 30.5 38.8 34.9
FreD 56.2 31.0 33.4   - 33.3 42.7 39.3
DDiF 69.2 42.0 45.3   - 45.8 64.6 53.4


GSDD **74.6** **43.4** **52.0** **69.4** **48.2** **73.6** **60.2**



**378**

**379**


**380**

**381**

**382**

**383**

**384**

**385**


**386**

**387**

**388**

**389**

**390**

**391**


**392**

**393**

**394**

**395**

**396**


**397**

**398**

**399**

**400**

**401**

**402**


**403**

**404**

**405**

**406**

**407**

**408**


**409**

**410**

**411**

**412**

**413**

**414**


**415**

**416**

**417**

**418**

**419**


**420**

**421**

**422**

**423**

**424**

**425**


**426**

**427**

**428**

**429**

**430**

**431**



Under review as a conference paper at ICLR 2026


Table 1: Test accuracy on ImageNet subsets (128 _×_
128) with trajectory matching (TM).


Subset Nette Woof Fruit Yellow Meow Squawk


Original 87.4 67.0 63.9 84.4 66.7 87.5


Input sized TM (Vanilla) 51.4 29.7 28.8 47.5 33.3 41.0

FRePo 48.1 29.7         -         -         -         

Static IDC 61.4 34.5 38.0 56.5 39.5 50.2

FreD 66.8 38.3 43.7 63.2 43.2 57.0


Parameterized HaBa 51.9 32.4 34.7 50.4 36.9 41.9

SPEED 66.9 38.0 43.4 62.6 43.6 60.9

NSD 68.6 35.2 39.8 61.0 45.2 52.9


Generative Prior GLaD 38.7 23.4 23.1  - 26.0 35.8

H-GLaD 45.4 28.3 25.6        - 29.6 39.7


Function DDiF 72.0 42.9 48.2 69.0 47.4 67.0


Primitive GSDD **76.4 46.4 51.2** **72.4** **53.8** **73.2**



SPEED Wei et al. (2023), NSD Yang et al. (2025), GLaD Cazenavette et al. (2023), H-GLaD Zhong
et al. (2025), LD3M Moser et al. (2024) and DDiF Shin et al. (2025).


**Experimental Details** We use TM Cazenavette et al. (2022), DM Zhao & Bilen (2023), and
DC Zhao et al. (2021) as the underlying distillation algorithms in our experiments. All parameters
of Gaussians are trained using the Adam optimizer, and the learning rate for Gaussian parameters is
uniformly set to 0.001. For CIFAR datasets, we apply ZCA whitening as a standard preprocessing
step. We fix the boundary loss weight to _λ_ boundary = 0 _._ 1. Experiments are conducted on NVIDIA
V100 and RTX 4090 GPUs. To initialize the Gaussian representation, we randomly sample real images from the original dataset and fit them with Gaussians before entering the standard distillation
loop. We provide complete implementation details and hyperparameters in the Appendix A.


**Storage Budget and Evaluation Protocol** We evaluate performance under varying storage budgets, specified in IPC (images per class). Since our method utilizes multiple low-storage Gaussian
images, we additionally report GPC (Gaussian Images per Class) in Appendix A. The number of
Gaussians per image is computed as pts = [res] _[×]_ [res] GPC _[×]_ [3] _×_ _[×]_ 9 [IPC] _[×]_ [2], where the factor of 2 accounts for the

use of bf16, and the denominator 9 reflects the number of parameters per Gaussian.


**Performance Comparison** Through the integration of GSDD, our method consistently outperforms the pixel-based TM baseline and other dataset parameterization methods on six ImageNet
subsets with IPC=1, as shown in Table 1. The results on the higher-resolution ImageNet subset
(256×256) are provided in Table 3, where our method remains superior at higher resolutions. In
addition, we also report experiments conducted at a lower resolution on CIFAR-10/100, with the
results presented in Appendix Table 15. When compared to the state-of-the-art method DDiF, our
approach achieves consistent improvements, demonstrating highly competitive performance. These
results highlight the effectiveness of GSDD in enhancing the quality of distilled datasets. We further
report results on ImageNet-1K(224 _×_ 224) in Table 18 in Appendix B.4.


**Universality to Matching Objectives** GSDD is designed to be compatible with a wide range
of dataset distillation algorithms and can be directly integrated with them. We evaluate GSDD
under two widely adopted loss paradigms: gradient matching (DC) and distribution matching (DM).
Experimental results, as shown in Table 2, confirm that GSDD robustly enhances performance across
different distillation objectives.


**Cross-Architecture Performance** An essential property of dataset distillation is robustness across
model architectures. The distilled data should retain high performance even when the evaluation
model differs from the one used during distillation. To assess this, we synthesize data using a
ConvNet and evaluate it on a variety of downstream architectures, including ResNet, VGG, AlexNet
and ViT. The results, summarized in Table 4 and Table 16, demonstrate that the increased diversity
introduced by GSDD leads to substantial improvements in cross-architecture generalization.


8


Under review as a conference paper at ICLR 2026











**432**

**433**


**434**

**435**

**436**

**437**

**438**

**439**


**440**

**441**

**442**

**443**

**444**

**445**


**446**

**447**

**448**

**449**

**450**


**451**

**452**

**453**

**454**

**455**

**456**


**457**

**458**

**459**

**460**

**461**

**462**


**463**

**464**

**465**

**466**

**467**

**468**


**469**

**470**

**471**

**472**

**473**


**474**

**475**

**476**

**477**

**478**

**479**


**480**

**481**

**482**

**483**

**484**

**485**


|Forward (B<br>res<br>102<br>(ms)<br>101<br>Time<br>100<br>Fw<br>85 MB 171<br>0 100 200|Col2|Col3|Col4|
|---|---|---|---|
|~~0~~<br>~~100~~<br>~~200~~<br> <br>100<br>101<br>102<br>Time (ms)<br>Forward (B<br>res<br>~~Fw~~<br>85 MB<br>171||||
|~~0~~<br>~~100~~<br>~~200~~<br> <br>100<br>101<br>102<br>Time (ms)<br>Forward (B<br>res<br>~~Fw~~<br>85 MB<br>171||||
|~~0~~<br>~~100~~<br>~~200~~<br> <br>100<br>101<br>102<br>Time (ms)<br>Forward (B<br>res<br>~~Fw~~<br>85 MB<br>171||||
|~~0~~<br>~~100~~<br>~~200~~<br> <br>100<br>101<br>102<br>Time (ms)<br>Forward (B<br>res<br>~~Fw~~<br>85 MB<br>171|||~~ Mem (MB)~~<br> MB<br>256 MB<br>341|
|~~0~~<br>~~100~~<br>~~200~~<br> <br>100<br>101<br>102<br>Time (ms)<br>Forward (B<br>res<br>~~Fw~~<br>85 MB<br>171|||~~300~~<br>~~400~~<br>~~5~~<br> <br> <br> <br>|


|Fwd+Bwd (Batch Sw<br>res=128|Col2|
|---|---|
|||
|||
|||
|3917|~~FB Mem (MB)~~<br> MB<br>7833 MB<br>11750 MB|
|~~0~~<br><br>|~~100~~<br>~~200~~<br>~~300~~<br><br> <br> <br> <br>|


|Col1|Res Sweep)|Col3|
|---|---|---|
||es Sweep)<br>2|es Sweep)<br>2|
||||
||||
||||
|~~FB Mem~~<br> B<br>7831 MB|~~  MB)~~<br>11747 MB<br>15662|~~00~~<br> B|
|~~0~~<br>~~200~~<br><br><br> <br>|~~300~~<br>~~400~~<br>~~5~~<br><br> <br>|~~300~~<br>~~400~~<br>~~5~~<br><br> <br>|



To evaluate the efficiency and representational capacity of GSDD, we conduct an image-fitting experiment comparing several recent parameterization methods with the same storage budget, including FreD Shin et al. (2023), SPEED Wei et al. (2023) and DDiF Shin et al. (2025). Specifically, we
use the ImageNette dataset and fit the first 100 images from each class (1,000 images in total) at a
resolution of 128. The optimization is performed for 1,000 steps, and for each method we perform
a grid search over learning rates and optimizers to ensure a fair comparison. As shown in Table 5,
GSDD shows great performance across representational quality (PSNR), GPU memory usage, and
runtime efficiency. High fidelity and low computational overhead of GSDD enables fast synthesis of
more diverse (Higher GPC) distilled datasets, which in turn leads to improved downstream performance. We further verify in Section 4.3 that larger GPC values with fixed IPC generally correspond
to stronger distilled data performance.


We further conduct a comprehensive quantitative comparison with the state-of-the-art DDiF method in terms Table 5: Comparison of Parameterization
of both decoding and training performance. As shown Methods on Image Fitting Efficiency and
in Figure 5, we measure the inference time and memory Representational Quality
consumption of both methods across varying per-image
storage budgets, image resolutions, and batch sizes. The FreD SPEED DDiF GSDD
results demonstrate that GSDD consistently outperforms

PSNR(dB)↑ 15.69 17.77 **23.23** 23.22

DDiF in both inference/training speed and memory ef- VRAM(MB)↓ 280.4 298.5 1078.8 **150.2**
ficiency. This advantage becomes especially pronounced Time(s)↓ **15.5** 69.4 1088.4 24.5
when handling high-resolution inputs or large batch sizes,
where GSDD achieves several-fold, or even an order-ofmagnitude, reductions in both computation time and memory usage. These findings indicate that our
method not only maintains strong image representation capabilities for distillation but also offers
superior rendering speed and significantly lower memory footprint, thereby substantially improving
the scalability of dataset distillation methods in practical deployment scenarios.





















Figure 5: Performance comparison between GSDD and DDiF under the same per-image storage budget (abbreviated as _st_ (floats)). Top row: Forward and forward+backward execution time and memory usage across
varying image resolutions (with fixed batch size = 32). Bottom row: Same metrics across varying batch sizes
(with fixed resolution = 128). GSDD consistently achieves lower latency and memory consumption, especially
under high-resolution and large-batch scenarios.



Table 3: Results on ImageNet-subsets with Distribution Matching (DM) for 256 _×_ 256


Method Nette Woof Fruit Yellow Meow Squawk


Vanilla 32.1 20.0 19.5 33.4 21.2 27.6
IDC 53.7 30.2 33.1 52.2 34.6 47.0
FreD 54.2 31.2 32.5 49.1 34.0 43.1
LatentDD 56.1 28.0 30.7  - 36.3 47.1
DDiF 67.8 39.6 43.2 63.1 44.8 67.0
GSDD **70.0 42.6 51.2** **67.4** **46.4** **70.4**


4.2 BENCHMARK RENDERING EFFICIENCY



Table 4: Cross-Architecture Performance


Subset Nette Woof Fruit Yellow Meow Squawk


TM 22.0 14.8 17.1 22.3 16.2 25.5
IDC 27.9 19.5 23.9 28.0 19.8 29.9
FreD 36.2 23.7 23.6 31.2 19.1 37.4
GLaD 30.4 17.1 21.1 - 19.6 28.2
H-GLaD 30.8 17.4 21.5 - 20.1 28.8
LD3M 32.0 19.9 21.4 - 22.1 30.4
DDiF **59.3** 34.1 39.3 51.1 33.8 54.0
GSDD 58.1 **34.6 39.9** **53.6** **34.0** **58.0**



Table 5: Comparison of Parameterization
Methods on Image Fitting Efficiency and
Representational Quality



FreD SPEED DDiF GSDD



PSNR(dB)↑ 15.69 17.77 **23.23** 23.22
VRAM(MB)↓ 280.4 298.5 1078.8 **150.2**
Time(s)↓ **15.5** 69.4 1088.4 24.5



9


Under review as a conference paper at ICLR 2026











**486**

**487**


**488**

**489**

**490**

**491**

**492**

**493**


**494**

**495**

**496**

**497**

**498**

**499**


**500**

**501**

**502**

**503**

**504**


**505**

**506**

**507**

**508**

**509**

**510**


**511**

**512**

**513**

**514**

**515**

**516**


**517**

**518**

**519**

**520**

**521**

**522**


**523**

**524**

**525**

**526**

**527**


**528**

**529**

**530**

**531**

**532**

**533**


**534**

**535**

**536**

**537**

**538**

**539**


|80 IPC=10, TM|Col2|Col3|IPC=1|10, TM|
|---|---|---|---|---|
|~~1~~<br>~~10 40~~<br>~~80~~<br>~~1~~<br>GPC<br>20<br>40<br>60<br>80<br>Accuracy (%)<br>|||||
|~~1~~<br>~~10 40~~<br>~~80~~<br>~~1~~<br>GPC<br>20<br>40<br>60<br>80<br>Accuracy (%)<br>|||||
|~~1~~<br>~~10 40~~<br>~~80~~<br>~~1~~<br>GPC<br>20<br>40<br>60<br>80<br>Accuracy (%)<br>|~~1~~|~~0 4~~|<br>~~80~~<br>~~1~~|<br>~~80~~<br>~~1~~|


|80 IPC=1, TM|Col2|IPC=1|1, TM|Col5|
|---|---|---|---|---|
|~~4 16~~<br>~~64~~<br>~~128~~<br>GPC<br>30<br>40<br>50<br>60<br>70<br>0<br>|||||
|~~4 16~~<br>~~64~~<br>~~128~~<br>GPC<br>30<br>40<br>50<br>60<br>70<br>0<br>|||||
|~~4 16~~<br>~~64~~<br>~~128~~<br>GPC<br>30<br>40<br>50<br>60<br>70<br>0<br>|~~ 1~~|~~ 6~~<br>~~6~~|~~1~~|~~8~~|


|70 IPC=1, TM|Col2|IPC|C=1|1, TM|Col6|Col7|
|---|---|---|---|---|---|---|
|~~20 30 40 50 60~~<br>~~80~~<br>~~100~~<br>GPC<br>40<br>50<br>60<br>0<br>|||||||
|~~20 30 40 50 60~~<br>~~80~~<br>~~100~~<br>GPC<br>40<br>50<br>60<br>0<br>|||||||
|~~20 30 40 50 60~~<br>~~80~~<br>~~100~~<br>GPC<br>40<br>50<br>60<br>0<br>||~~  0 5~~|~~   0 60~~<br>~~80~~<br>~~1~~|~~   0 60~~<br>~~80~~<br>~~1~~|~~   0 60~~<br>~~80~~<br>~~1~~|~~00~~|


|5<br>0|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|~~100 120~~<br>~~16~~<br><br>5|~~00 1~~|~~ 20~~<br>~~16~~|~~200~~<br>~~2~~|~~200~~<br>~~2~~|~~0~~|
|~~100 120~~<br>~~16~~<br><br>5|~~00 1~~|~~ 20~~<br>~~16~~|PC|PC|PC|



We further examine the effect of GPC through an ablation study, as shown in Figure 6. When
using the memory-efficient DM algorithm or when distilling under small storage budgets, performance increases with GPC at first but eventually declines. This is because small GPC values lead
to insufficient diversity, whereas excessively large GPC values allocate too few Gaussians per image, weakening each image’s representational capacity. In more memory-consuming settings, the
optimal turning point of GPC becomes difficult to reach, and performance tends to improve monotonically as GPC increases. Figure 6 also shows that the performance of the initial dataset serves as
a useful reference for estimating a reasonable GPC range.


5 CONCLUSION


This work introduces sparse Gaussian representations into the task of dataset distillation parameterization. We propose a complete framework for encoding distilled images using 2D Gaussian
primitives, including the underlying mathematical formulation, a parallel differentiable renderer,
and several tailored components to handle the sparsity of distilled datasets. Specifically, we incorporate anti-aliasing and spatial constraints to enhance the utility of Gaussians and improve overall
distillation quality. Through extensive quantitative and qualitative experiments, we demonstrate
that sparse Gaussian representations facilitate more effective optimization, offer strong scalability,
and provide better coverage across samples of varying difficulty. As shown in our results, GSDD
achieves highly competitive performance under various datasets and storage budgets, while also
exhibiting superior computational efficiency compared to DDiF. The proposed method is simple,
plug-and-play, and readily compatible with existing distillation algorithms, leaving ample room for
future extensions. Future work may focus on scaling to larger datasets, extending Gaussian representations to video modalities, and exploring dynamic density control of Gaussian primitives to
further enhance representation quality.


10















Figure 6: Effect of GPC (Gaussian Images Per Class) on distilled dataset performance across different datasets, storage budgets, and distillation algorithms (TM Cazenavette et al. (2022) and DM Zhao
& Bilen (2023)). Initial Accuracy denotes the performance of initialized gaussian images (initialized
on sampled real images). Distilled Accuracy denotes the performance after distillation.


4.3 ABLATION STUDY



To investigate the contribution of each component to model performance,
we conduct a series of ablation studies, with results summarized in Table 6. The experiments show that opacity modeling and bf16 mixedprecision training are two major factors, as removing either leads to a
notable performance drop. The boundary entry in the table refers to the
result obtained when both hard clipping and loss regularization are removed. The impact of anti-aliasing exhibits a dependence on the dataset
and the complexity of the representation. Its effect becomes important
when the number of Gaussians used to represent each image is limited.
Specifically, for datasets like CIFAR-10, where storage budgets are constrained and each image is represented by a small number of Gaussians,
disabling anti-aliasing introduces rendering artifacts that degrade performance. In contrast, for datasets like ImageNette with denser Gaussians,
aliasing artifacts are largely mitigated, and the performance gains from
anti-aliasing become relatively minor.



Table 6: Ablation Study
on Individual Components


ImageNette ACC


w/o all 74.2
w/o opacity 74.8
w/o boundary 75.1
w/o bf16 74.8
w/o antialias 76.4
all **76.4**


CIFAR-10 ACC
w/o antialias 74.5
all **75.5**


**540**

**541**


**542**

**543**

**544**

**545**

**546**

**547**


**548**

**549**

**550**

**551**

**552**

**553**


**554**

**555**

**556**

**557**

**558**


**559**

**560**

**561**

**562**

**563**

**564**


**565**

**566**

**567**

**568**

**569**

**570**


**571**

**572**

**573**

**574**

**575**

**576**


**577**

**578**

**579**

**580**

**581**


**582**

**583**

**584**

**585**

**586**

**587**


**588**

**589**

**590**

**591**

**592**

**593**



Under review as a conference paper at ICLR 2026


REPRODUCIBILITY STATEMENT


We provide a comprehensive description of our method in Section 3, and detailed hyperparameters,
optimizers, and network architectures for each dataset in Appendix A. All experiments are conducted
on publicly available datasets. Our core implementation has been submitted to the supplemental
material and more detailed codebase will be open-sourced upon acceptance of the paper.


ETHICS STATEMENT


This work complies with the ICLR Code of Ethics. It does not involve any unethical experimentation, nor does it use private, sensitive, or personally identifiable data. All datasets employed in
our experiments are publicly available and properly licensed. No data was collected from human
subjects or required approval from an institutional review board.


Our research promotes responsible stewardship of machine learning technology. In particular, by
advancing the use of synthetic representations for dataset distillation, our approach supports privacypreserving research and reduces the reliance on real-world data that might raise privacy concerns.


We have taken care to ensure transparency and reproducibility. All implementation details, including
hyperparameters, architectures, and code necessary for reproduction, are provided in the supplemental materials and will be open-sourced upon acceptance. Furthermore, we acknowledge and cite all
prior works that our approach builds upon, respecting intellectual contributions in line with academic
and ethical standards.


LLM USAGE STATEMENT


All research components presented in this paper, including the initial literature review, problem
formulation, methodological development, and experimental design, were independently conceived
and developed by the authors, without any contribution from large language models (LLMs).


LLMs were used solely as general-purpose tools for non-substantive assistance. Specifically, we
employed LLMs to help polish the writing, check grammar, and clarify LaTeX syntax during the
preparation of the manuscript. In all such cases, the authors critically reviewed, verified, and revised
the generated content to ensure its accuracy and originality. The authors take full responsibility for
the final text, including any parts informed by LLM assistance.


In addition, we made limited use of AI-assisted coding tools (e.g., GitHub Copilot in VSCode) for
basic code autocompletion during code implementation. LLMs were also used to assist in initial
framework scaffolding for code visualization and to suggest potential debugging strategies. All core
algorithmic components and experimental pipelines were manually written, verified, and refined by
the authors. Every LLM-assisted suggestion was critically assessed and edited to ensure correctness
and alignment with the intended methodology.


LLMs were not used to generate, modify, or interpret any experimental results or conclusions.


REFERENCES


George Cazenavette, Tongzhou Wang, Antonio Torralba, Alexei A. Efros, and Jun-Yan Zhu. Dataset
distillation by matching training trajectories. In _Proceedings of the IEEE/CVF Conference on_
_Computer Vision and Pattern Recognition (CVPR)_, pp. 10718–10727, June 2022.


George Cazenavette, Tongzhou Wang, Antonio Torralba, Alexei A. Efros, and Jun-Yan Zhu. Generalizing dataset distillation via deep generative prior. In _Proceedings of the IEEE/CVF Conference_
_on Computer Vision and Pattern Recognition (CVPR)_, pp. 3739–3748, June 2023.


Du Chen, Liyi Chen, Zhengqiang Zhang, and Lei Zhang. Generalized and efficient 2d gaussian
splatting for arbitrary-scale super-resolution, 2025.


Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xiaofeng Yang, Yikai Wang, Zhongang Cai, Lei
Yang, Huaping Liu, and Guosheng Lin. Gaussianeditor: Swift and controllable 3d editing with


11


**594**

**595**


**596**

**597**

**598**

**599**

**600**

**601**


**602**

**603**

**604**

**605**

**606**

**607**


**608**

**609**

**610**

**611**

**612**


**613**

**614**

**615**

**616**

**617**

**618**


**619**

**620**

**621**

**622**

**623**

**624**


**625**

**626**

**627**

**628**

**629**

**630**


**631**

**632**

**633**

**634**

**635**


**636**

**637**

**638**

**639**

**640**

**641**


**642**

**643**

**644**

**645**

**646**

**647**



Under review as a conference paper at ICLR 2026


gaussian splatting. In _Proceedings of the IEEE/CVF conference on computer vision and pattern_
_recognition_, pp. 21476–21485, 2024.


Ming-Yu Chung, Sheng-Yen Chou, Chia-Mu Yu, Pin-Yu Chen, Sy-Yen Kuo, and Tsung-Yi Ho.
Rethinking backdoor attacks on dataset distillation: A kernel method perspective. In _The Twelfth_
_International Conference on Learning Representations_, 2024.


Justin Cui, Ruochen Wang, Si Si, and Cho-Jui Hsieh. Scaling up dataset distillation to imagenet1k with constant memory. In _Proceedings of the 40th International Conference on Machine_
_Learning_, volume 202 of _Proceedings of Machine Learning Research_, pp. 6565–6590, July 2023.


Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In _2009 IEEE Conference on Computer Vision and Pattern Recognition_,
pp. 248–255, 2009.


Zhiwei Deng and Olga Russakovsky. Remember the past: Distilling datasets into addressable memories for neural networks. In _Advances in Neural Information Processing Systems_, volume 35, pp.
34391–34404, 2022.


Tian Dong, Bo Zhao, and Lingjuan Lyu. Privacy for free: How does dataset condensation help
privacy? In _Proceedings of the 39th International Conference on Machine Learning_, volume 162
of _Proceedings of Machine Learning Research_, pp. 5378–5396, July 2022.


Jiawei Du, Yidi Jiang, Vincent Y. F. Tan, Joey Tianyi Zhou, and Haizhou Li. Minimizing the
accumulated trajectory error to improve dataset distillation. In _Proceedings of the IEEE/CVF_
_Conference on Computer Vision and Pattern Recognition (CVPR)_, pp. 3749–3758, June 2023a.


Jiawei Du, Qin Shi, and Joey Tianyi Zhou. Sequential subset matching for dataset distillation. In
_Advances in Neural Information Processing Systems_, volume 36, pp. 67487–67504, 2023b.


Yunzhen Feng, Shanmukha Ramakrishna Vedantam, and Julia Kempe. Embarrassingly simple
dataset distillation. In _The Twelfth International Conference on Learning Representations_, 2024.


Jianyang Gu, Kai Wang, Wei Jiang, and Yang You. Summarizing stream data for memoryconstrained online continual learning. _Proceedings of the AAAI Conference on Artificial Intel-_
_ligence_, 38(11):12217–12225, March 2024.


Ziyao Guo, Kai Wang, George Cazenavette, HUI LI, Kaipeng Zhang, and Yang You. Towards
lossless dataset distillation via difficulty-aligned trajectory matching. In _The Twelfth International_
_Conference on Learning Representations_, 2024.


Jintong Hu, Bin Xia, Bin Chen, Wenming Yang, and Lei Zhang. Gaussiansr: High fidelity 2d
gaussian splatting for arbitrary-scale image super-resolution, 2024.


Chun-Yin Huang, Kartik Srinivas, Xin Zhang, and Xiaoxiao Li. Overcoming data and model heterogeneities in decentralized federated learning via synthetic anchors, 2024a. [URL https:](https://openreview.net/forum?id=PcBJ4pA6bF)
[//openreview.net/forum?id=PcBJ4pA6bF.](https://openreview.net/forum?id=PcBJ4pA6bF)


Nan Huang, Xiaobao Wei, Wenzhao Zheng, Pengju An, Ming Lu, Wei Zhan, Masayoshi Tomizuka,
Kurt Keutzer, and Shanghang Zhang. S3gaussian: Self-supervised street gaussians for autonomous driving. _arXiv preprint arXiv:2405.20323_, 2024b.


Yi-Hua Huang, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu, Yan-Pei Cao, and Xiaojuan Qi. Scgs: Sparse-controlled gaussian splatting for editable dynamic scenes. In _Proceedings of the_
_IEEE/CVF conference on computer vision and pattern recognition_, pp. 4220–4230, 2024c.


Yuqi Jia, Saeed Vahidian, Jingwei Sun, Jianyi Zhang, Vyacheslav Kungurtsev, Neil Zhenqiang
Gong, and Yiran Chen. Unlocking the potential of federated learning: The symphony of dataset
distillation via deep generative latents. In _Computer Vision – ECCV 2024_, pp. 18–33, Cham,
2025. ISBN 978-3-031-73229-4.


Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child,
Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language
[models, 2020. URL https://arxiv.org/abs/2001.08361.](https://arxiv.org/abs/2001.08361)


12


**648**

**649**


**650**

**651**

**652**

**653**

**654**

**655**


**656**

**657**

**658**

**659**

**660**

**661**


**662**

**663**

**664**

**665**

**666**


**667**

**668**

**669**

**670**

**671**

**672**


**673**

**674**

**675**

**676**

**677**

**678**


**679**

**680**

**681**

**682**

**683**

**684**


**685**

**686**

**687**

**688**

**689**


**690**

**691**

**692**

**693**

**694**

**695**


**696**

**697**

**698**

**699**

**700**

**701**



Under review as a conference paper at ICLR 2026


Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. _ACM Trans. Graph._, 42(4), July 2023. ISSN 07300301.


Jang-Hyun Kim, Jinuk Kim, Seong Joon Oh, Sangdoo Yun, Hwanjun Song, Joonhyun Jeong, JungWoo Ha, and Hyun Oh Song. Dataset condensation via efficient synthetic-data parameterization.
In _Proceedings of the 39th International Conference on Machine Learning_, volume 162 of _Pro-_
_ceedings of Machine Learning Research_, pp. 11102–11118, July 2022.


Saehyung Lee, Sanghyuk Chun, Sangwon Jung, Sangdoo Yun, and Sungroh Yoon. Dataset condensation with contrastive signals. In _Proceedings of the 39th International Conference on Machine_
_Learning_, volume 162 of _Proceedings of Machine Learning Research_, pp. 12352–12364, July
2022.


Dai Liu, Jindong Gu, Hu Cao, Carsten Trinitis, and Martin Schulz. Dataset distillation by automatic
training trajectories. In _Computer Vision – ECCV 2024_, pp. 334–351, Cham, 2025. ISBN 978-3031-73021-4.


Songhua Liu and Xinchao Wang. Mgdd: A meta generator for fast dataset distillation. In _Advances_
_in Neural Information Processing Systems_, volume 36, pp. 56437–56455, 2023.


Songhua Liu, Kai Wang, Xingyi Yang, Jingwen Ye, and Xinchao Wang. Dataset distillation via
factorization. In _Advances in Neural Information Processing Systems_, volume 35, pp. 1100–1113,
2022.


Yanqing Liu, Jianyang Gu, Kai Wang, Zheng Zhu, Wei Jiang, and Yang You. Dream: Efficient
dataset distillation by representative matching. In _Proceedings of the IEEE/CVF International_
_Conference on Computer Vision (ICCV)_, pp. 17314–17324, October 2023.


Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and
Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. _Communications_
_of The Acm_, 65(1):99–106, December 2021. ISSN 0001-0782.


Brian B. Moser, Federico Raue, Sebastian Palacio, Stanislav Frolov, and Andreas Dengel. Latent
[dataset distillation with diffusion models, 2024. URL https://arxiv.org/abs/2403.](https://arxiv.org/abs/2403.03881)
[03881.](https://arxiv.org/abs/2403.03881)


Zhiyin Qian, Shaofei Wang, Marko Mihajlovic, Andreas Geiger, and Siyu Tang. 3dgs-avatar: Animatable avatars via deformable 3d gaussian splatting. In _Proceedings of the IEEE/CVF conference_
_on computer vision and pattern recognition_, pp. 5020–5030, 2024.


Donghyeok Shin, Seungjae Shin, and Il-chul Moon. Frequency domain-based dataset distillation.
In _Advances in Neural Information Processing Systems_, volume 36, pp. 70033–70044, 2023.


Donghyeok Shin, HeeSun Bae, Gyuwon Sim, Wanmo Kang, and Il-chul Moon. Distilling dataset
into neural field. In _The Thirteenth International Conference on Learning Representations_, 2025.


Vincent Sitzmann, Julien Martel, Alexander Bergman, David Lindell, and Gordon Wetzstein. Implicit neural representations with periodic activation functions. In _Advances in Neural Information_
_Processing Systems_, volume 33, pp. 7462–7473, 2020.


Felipe Petroski Such, Aditya Rawal, Joel Lehman, Kenneth Stanley, and Jeffrey Clune. Generative teaching networks: Accelerating neural architecture search by learning to generate synthetic
training data. In _International Conference on Machine Learning_, pp. 9206–9216. PMLR, 2020.


Peng Sun, Bei Shi, Daiwei Yu, and Tao Lin. On the diversity and realism of distilled dataset: An
efficient dataset distillation paradigm. In _Proceedings of the IEEE/CVF Conference on Computer_
_Vision and Pattern Recognition (CVPR)_, pp. 9390–9399, June 2024.


Longan Wang, Yuang Shi, and Wei Tsang Ooi. Gsvc: Efficient video representation and compression
through 2d gaussian splatting, 2025a.


13


**702**

**703**


**704**

**705**

**706**

**707**

**708**

**709**


**710**

**711**

**712**

**713**

**714**

**715**


**716**

**717**

**718**

**719**

**720**


**721**

**722**

**723**

**724**

**725**

**726**


**727**

**728**

**729**

**730**

**731**

**732**


**733**

**734**

**735**

**736**

**737**

**738**


**739**

**740**

**741**

**742**

**743**


**744**

**745**

**746**

**747**

**748**

**749**


**750**

**751**

**752**

**753**

**754**

**755**



Under review as a conference paper at ICLR 2026


Shaobo Wang, Yicun Yang, Zhiyuan Liu, Chenghao Sun, Xuming Hu, Conghui He, and Linfeng
Zhang. Dataset distillation with neural characteristic function: A minmax perspective. In _Pro-_
_ceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_, pp.
25570–25580, June 2025b.


Tongzhou Wang, Jun-Yan Zhu, Antonio Torralba, and Alexei A. Efros. Dataset distillation. _CoRR_,
abs/1811.10959, 2018.


Wei Wei, Tom De Schepper, and Kevin Mets. Dataset condensation with latent quantile matching. In
_Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_
_Workshops_, pp. 7703–7712, June 2024.


Xing Wei, Anjia Cao, Funing Yang, and Zhiheng Ma. Sparse parameterization for epitomic dataset
distillation. In _Advances in Neural Information Processing Systems_, volume 36, pp. 50570–50596,
2023.


Sebastian Weiss and Derek Bradley. Gaussian billboards: Expressive 2d gaussian splatting with
textures, 2024.


Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian,
and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In _Proceedings_
_of the IEEE/CVF conference on computer vision and pattern recognition_, pp. 20310–20320, 2024.


Guochen Yan, Luyuan Xie, Xinyi Gao, Wentao Zhang, Qingni Shen, Yuejian Fang, and Zhonghai Wu. Fedvck: Non-iid robust and communication-efficient federated learning via valuable
condensed knowledge for medical image analysis. In _Proceedings of the AAAI Conference on_
_Artificial Intelligence_, volume 39, pp. 21904–21912, 2025.


Enneng Yang, Li Shen, Zhenyi Wang, Tongliang Liu, and Guibing Guo. An efficient dataset condensation plugin and its application to continual learning. In _Thirty-Seventh Conference on Neural_
_Information Processing Systems_, 2023.


Shaolei Yang, Shen Cheng, Mingbo Hong, Haoqiang Fan, Xing Wei, and Shuaicheng Liu. Neural
spectral decomposition for dataset distillation. In _Computer Vision – ECCV 2024_, pp. 275–290,
Cham, 2025. ISBN 978-3-031-72943-0.


Zeyuan Yin, Eric Xing, and Zhiqiang Shen. Squeeze, recover and relabel: Dataset condensation at
imagenet scale from a new perspective. In _Advances in Neural Information Processing Systems_,
volume 36, pp. 73582–73603, 2023.


Xinjie Zhang, Xingtong Ge, Tongda Xu, Dailan He, Yan Wang, Hongwei Qin, Guo Lu, Jing Geng,
and Jun Zhang. Gaussianimage: 1000 fps image representation and compression by 2d gaussian
splatting. In _Computer Vision – ECCV 2024_, pp. 327–345, Cham, 2025. ISBN 978-3-031-726736.


Bo Zhao and Hakan Bilen. Dataset condensation with distribution matching. In _Proceedings of_
_the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)_, pp. 6514–6523,
January 2023.


Bo Zhao, Konda Reddy Mopuri, and Hakan Bilen. Dataset condensation with gradient matching. In
_International Conference on Learning Representations_, 2021.


Ganlong Zhao, Guanbin Li, Yipeng Qin, and Yizhou Yu. Improved distribution matching for dataset
condensation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern_
_Recognition (CVPR)_, pp. 7856–7865, June 2023.


Runkai Zheng, Vishnu Asutosh Dasu, Yinong Oliver Wang, Haohan Wang, and Fernando De la
Torre. Improving noise efficiency in privacy-preserving dataset distillation, 2025.


Xinhao Zhong, Hao Fang, Bin Chen, Xulin Gu, Meikang Qiu, Shuhan Qi, and Shu-Tao Xia. Hierarchical features matter: A deep exploration of progressive parameterization method for dataset
distillation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recog-_
_nition (CVPR)_, pp. 30462–30471, June 2025.


14


**756**

**757**


**758**

**759**

**760**

**761**

**762**

**763**


**764**

**765**

**766**

**767**

**768**

**769**


**770**

**771**

**772**

**773**

**774**


**775**

**776**

**777**

**778**

**779**

**780**


**781**

**782**

**783**

**784**

**785**

**786**


**787**

**788**

**789**

**790**

**791**

**792**


**793**

**794**

**795**

**796**

**797**


**798**

**799**

**800**

**801**

**802**

**803**


**804**

**805**

**806**

**807**

**808**

**809**



Under review as a conference paper at ICLR 2026


Xiaoyu Zhou, Zhiwei Lin, Xiaojun Shan, Yongtao Wang, Deqing Sun, and Ming-Hsuan Yang. Drivinggaussian: Composite gaussian splatting for surrounding dynamic autonomous driving scenes.
In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_, pp.
21634–21643, 2024.


Yongchao Zhou, Ehsan Nezhadarya, and Jimmy Ba. Dataset distillation using neural feature regression. In _Advances in Neural Information Processing Systems_, volume 35, pp. 9813–9827,
2022.


Lingting Zhu, Guying Lin, Jinnan Chen, Xinjie Zhang, Zhenchao Jin, Zhao Wang, and Lequan Yu.
Large images are gaussians: High-quality large image representation with levels of 2d gaussian
splatting, 2025.


Wojciech Zielonka, Timur Bagautdinov, Shunsuke Saito, Michael Zollh¨ofer, Justus Thies, and Javier
Romero. Drivable 3d gaussian avatars. In _2025 International Conference on 3D Vision (3DV)_,
pp. 979–990. IEEE, 2025.


15


**810**

**811**


**812**

**813**

**814**

**815**

**816**

**817**


**818**

**819**

**820**

**821**

**822**

**823**


**824**

**825**

**826**

**827**

**828**


**829**

**830**

**831**

**832**

**833**

**834**


**835**

**836**

**837**

**838**

**839**

**840**


**841**

**842**

**843**

**844**

**845**

**846**


**847**

**848**

**849**

**850**

**851**


**852**

**853**

**854**

**855**

**856**

**857**


**858**

**859**

**860**

**861**

**862**

**863**



Under review as a conference paper at ICLR 2026


A EXPERIMENTAL DETAILS


A.1 DATASETS


We conduct experiments on eight datasets in total, including CIFAR-10, CIFAR-100, and six ImageNet subsets: _imagenette_, _imagewoof_, _imagefruit_, _imagemeow_, _imagesquawk_, and _imageyellow_ .
CIFAR-10 consists of 60,000 images of resolution 32 _×_ 32 across 10 classes, with 50,000 training
images and 10,000 test images. CIFAR-100 has the same resolution and number of images, but
covers 100 classes. Each ImageNet subset contains images of resolution 128 _×_ 128 from 10 classes,
with more than 10,000 images per subset.


A.2 NETWORK ARCHITECTURES


We adopt ConvNet architectures as the backbone networks. A Depth- _n_ ConvNet consists of _n_ blocks
followed by a fully-connected layer, where each block is composed of a 3 _×_ 3 convolutional layer
with 128 filters, instance normalization, a ReLU activation, and 2 _×_ 2 average pooling with stride 2.
Specifically, we use ConvNetD3 for CIFAR-10 and CIFAR-100, and ConvNetD5 for the ImageNet
subsets.


A.3 IMAGE INITIALIZATION


For the initialization of synthetic images, all experiments adopt the mean squared error (MSE) loss
and use the Adam optimizer with a learning rate of 1 _×_ 10 _[−]_ [2] .


A.4 DATASET DISTILLATION


During dataset distillation, Gaussian points are optimized with the Adam optimizer and a learning
rate of 1 _×_ 10 _[−]_ [2] . The number of optimization iterations is set to 15,000 unless otherwise specified.


Table 7: Hyperparameters for dataset distillation on CIFAR-10 32 _×_ 32.


Setting GPC num ~~p~~ oints syn ~~s~~ teps max ~~s~~ tart ~~e~~ poch expert ~~e~~ pochs lr ~~l~~ r lr ~~i~~ nit batch ~~s~~ yn zca


IPC=1 30 22 60 30 2 1e-5 1e-2 300 True
IPC=10 160 42 60 30 2 1e-5 1e-2 380 True
IPC=50 250 136 60 30 2 1e-5 1e-2 360 True


Table 8: Hyperparameters for dataset distillation on CIFAR-100 32 _×_ 32.


Setting GPC num ~~p~~ oints syn steps max start epoch expert epochs lr ~~l~~ r lr init batch syn zca


IPC=1 30 22 60 30 2 1e-5 1e-2 512 True
IPC=10 80 85 60 30 2 1e-5 1e-2 512 True
IPC=50 400 85 60 30 2 1e-5 1e-2 640 True


Table 9: Hyperparameters for dataset distillation on ImageNet-subset TM 128 _×_ 128.


Setting GPC num ~~p~~ oints syn ~~s~~ teps max ~~s~~ tart ~~e~~ poch expert ~~e~~ pochs lr ~~l~~ r lr ~~i~~ nit batch syn zca


IPC=1 64 170 20 40 2 1e-6 1e-2 150 False
IPC=10 640 170 20 40 2 1e-5 1e-2 160 False


A.5 PERFORMANCE BENCHMARKING


In Table 5, we evaluate the image-fitting performance of FreD Shin et al. (2023), SPEED Wei et al.
(2023), DDiF Shin et al. (2025), and GSDD. We select 100 images from each class of the ImageNette
dataset, resulting in a total of 1,000 images, all at a resolution of 128. Each method is constrained
to the same storage budget, meaning that, on average, each image is represented using 963 floatingpoint parameters. For SPEED, we follow the parameter-counting formula provided in its original


16


**864**

**865**


**866**

**867**

**868**

**869**

**870**

**871**


**872**

**873**

**874**

**875**

**876**

**877**


**878**

**879**

**880**

**881**

**882**


**883**

**884**

**885**

**886**

**887**

**888**


**889**

**890**

**891**

**892**

**893**

**894**


**895**

**896**

**897**

**898**

**899**

**900**


**901**

**902**

**903**

**904**

**905**


**906**

**907**

**908**

**909**

**910**

**911**


**912**

**913**

**914**

**915**

**916**

**917**



Under review as a conference paper at ICLR 2026


Table 10: Hyperparameters for dataset distillation on ImageNet-subset 128 _×_ 128 (DM).


Setting GPC num ~~p~~ oints batch ~~r~~ eal batch syn zca Iteration


IPC=1 200 54 1024 2000 False 20000


Table 11: Hyperparameters for dataset distillation on ImageNet-subset 256 _×_ 256 (DM).


Setting GPC num ~~p~~ oints batch ~~r~~ eal batch syn zca Iteration


IPC=1 120 364 512 512 False 20000


Table 12: Hyperparameters for dataset distillation on ImageNet-subset 128 _×_ 128 (DC).


Setting GPC num ~~p~~ oints batch ~~r~~ eal batch syn zca Iteration


IPC=1 200 54 720 2000 False 5000


Table 13: Performance gains of individual components on ImageNette.


w/ none w/ boundary w/ bf16 w/ opacity w/ antilias


74.2 74.6 76.0 76.0 74.4


Table 14: Fine-grained ablation on boundary constraints. ‘reg’ denotes the boundary loss regularization defined in Eq. 10, and ‘clip’ denotes the clipping operation defined in Eq. 11.


w/o reg w/o clip w/o reg+clip w/ reg+clip


76.2 76.2 75.1 76.4


paper,
#Params = _DK_ + 1 _._ 5 _NHk_ + _R_ (3 _D_ [2] 2 + 7 _D_ ) + _L_ ( _D_ + 1) _,_


and, given the storage constraint, we solve for all feasible hyperparameter combinations. The final configuration used in our experiments is (D=32, N=32, k=135, R=2, L=64). We attempted to
evaluate HaBa Liu et al. (2022) as well, but its model structure cannot be solved inversely under
the resolution of 128 and the storage budget of 963 parameters. FreD, DDiF, and GSDD exhibit
clear advantages in terms of estimating and meeting the storage budget. For each method, we search
over Adam and SGD optimizers and learning rates spaced exponentially by a factor of 10, and we
report the highest PSNR achieved. All models are trained for 1,000 iterations using MSE loss. Each
experiment is repeated three times with random initialization, and the average result is reported. The
results demonstrate that our method achieves strong representational capability and computational
efficiency.


B MORE EXPERIMENTAL RESULTS


B.1 PERFORMANCE COMPARISON ON LOW-DIMENSIONAL DATASETS


As shown in Table 15, our method GSDD consistently achieves the best performance across both
CIFAR-10 and CIFAR-100 under all IPC settings. Notably, GSDD surpasses all baselines by a clear
margin in the low-data regime (IPC=1), achieving 67.6% on CIFAR-10 and 43.0% on CIFAR-100.
In high-data settings (IPC=50), GSDD maintains its superiority, reaching 77.7% and 53.1% respectively. These results demonstrate the strong generalization ability and scalability of our primitivebased distillation framework.


B.2 DETAILED CROSS-ARCHITECTURE RESULTS


We evaluate the generalization ability of distilled data across different architectures, including AlexNet, VGG11, ResNet18, and ViT. The network implementations follow those provided


17


**918**

**919**


**920**

**921**

**922**

**923**

**924**

**925**


**926**

**927**

**928**

**929**

**930**

**931**


**932**

**933**

**934**

**935**

**936**


**937**

**938**

**939**

**940**

**941**

**942**


**943**

**944**

**945**

**946**

**947**

**948**


**949**

**950**

**951**

**952**

**953**

**954**


**955**

**956**

**957**

**958**

**959**


**960**

**961**

**962**

**963**

**964**

**965**


**966**

**967**

**968**

**969**

**970**

**971**



Under review as a conference paper at ICLR 2026


Table 15: Classification accuracy on CIFAR-10 and CIFAR-100 under different IPC settings.


CIFAR10 CIFAR100
Method

IPC=1 IPC=10 IPC=50 IPC=1 IPC=10 IPC=50


Input sized TM 46.3±0.8 65.3±0.7 71.6±0.2 24.3±0.3 40.1±0.4 47.7±0.2
FRePo 46.8±0.7 65.5±0.4 71.7±0.2 28.7±0.1 42.5±0.2 44.3±0.2


Static IDC 50.0±0.4 67.5±0.5 74.5±0.2  -  -  FreD 60.6±0.8 70.3±0.3 75.8±0.1 34.6±0.4 42.7±0.2 47.8±0.1


Parameterized HaBa 48.3±0.8 69.9±0.4 74.0±0.2  -  - 47.0±0.2
RTP 66.4±0.4 71.2±0.4 73.6±0.5 34.0±0.4 42.9±0.7       HMN 65.7±0.3 73.7±0.1 76.9±0.2 36.3±0.2 45.4±0.2 48.5±0.2
SPEED 63.2±0.1 73.5±0.2 77.7±0.4 40.4±0.4 45.9±0.3 49.1±0.2
NSD **68.5±0.8** 73.4±0.2 75.2±0.6 36.5±0.3 46.1±0.2       

Function DDiF 66.5±0.4 74.0±0.4 77.5±0.3 42.1±0.2 46.0±0.2 49.9±0.2


Primitive GSDD 67.6±0.4 **75.5±0.3** **77.7±0.5** **43.0±0.1** **47.4±0.3** **53.1±0.2**


in Cazenavette et al. (2023). For training, the learning rate is set to 1 _×_ 10 _[−]_ [3] for AlexNet, 5 _×_ 10 _[−]_ [5]
for ViT, and 1 _×_ 10 _[−]_ [2] for the remaining architectures. All models are optimized using Adam with a
cosine annealing learning rate schedule. We report the results averaged over three independent runs,
as shown in Figure 16.


Table 16: Detailed cross-architecture performance comparison.


Test Net Method Nette Woof Fruit Yellow Meow Squawk



B.3 MORE ABLATION STUDY


**Different initialization strategy** Prior methods often initialize distilled data using real images,
which may introduce additional privacy risks. We further conduct an ablation study on different
initialization strategies, including random initialization. However, we find that optimizing Gaussian
representations from purely random initialization is inherently unstable: the initial Gaussian points
are extremely small and sparsely distributed, creating large gaps that prevent effective gradient propagation. To mitigate this issue, we introduce a Solid-Color Warmup strategy. Instead of fitting any


18



AlexNet


VGG11


ResNet18


ViT



TM 13.2 ± 0.6 10.0 ± 0.0 10.0 ± 0.0 11.0 ± 0.2 9.8 ± 0.0 IDC 17.4 ± 0.9 16.5 ± 0.7 17.9 ± 0.7 20.6 ± 0.9 16.8 ± 0.5 20.7 ± 1.0
FreD 35.7 ± 0.4 23.9 ± 0.7 15.8 ± 0.7 19.8 ± 1.2 14.4 ± 0.5 36.3 ± 0.3
DDiF 60.7 ± 2.3 **36.4 ± 2.3** 41.8 ± 0.6 **56.2 ± 0.8** **40.3 ± 1.9** **60.5 ± 0.4**
GSDD **63.1 ± 1.3** 33.6 ± 0.9 **41.9 ± 2.0** 53.5 ± 0.9 38.1 ± 0.8 60.1 ± 0.2


TM 17.4 ± 2.1 12.6 ± 1.8 11.8 ± 1.0 16.9 ± 1.1 13.8 ± 1.3 IDC 19.6 ± 1.5 16.0 ± 2.1 13.8 ± 1.3 16.8 ± 3.5 13.1 ± 2.0 19.1 ± 1.2
FreD 21.8 ± 2.9 17.1 1.7 12.6 ± 2.6 18.2 ± 1.1 13.2 ± 1.9 18.6 ± 2.3
DDiF 53.6 ± 1.5 29.9 ± 1.9 33.8 ± 1.9 44.2 ± 1.7 **32.0 ± 1.8** 37.9 ± 1.5
GSDD **56.8 ± 1.6** **32.7 ± 1.4** **34.1 ± 3.0** **57.3 ± 4.5** 31.3 ± 1.8 **55.9 ± 3.4**


TM 34.9 ± 2.3 20.7 ± 1.0 23.1 ± 1.5 43.4 ± 1.1 22.8 ± 2.2 IDC 43.6 ± 1.3 23.2 ± 0.8 32.9 ± 2.8 44.2 ± 3.5 28.2 ± 0.5 47.8 ± 1.9
FreD 48.8 ± 1.8 28.4 ± 0.6 34.0 ± 1.9 49.3 ± 1.1 29.0 ± 1.8 50.2 ± 0.8
DDiF **63.8 ± 1.8** 37.5 ± 1.9 42.0 ± 1.9 **55.9 ± 1.0** 35.8 ± 1.8 **62.6 ± 1.5**
GSDD 59.1 ± 3.0 **39.5 ± 2.0** **42.6 ± 0.8** **55.9 ± 1.2** **40.9 ± 2.9** **62.6 ± 0.6**


TM 22.6 ± 1.1 15.9 ± 0.4 23.3 ± 0.4 18.1 ± 1.3 18.6 ± 0.9 IDC 31.0 ± 0.6 22.4 ± 0.8 31.1 ± 0.8 30.3 ± 0.6 21.4 ± 0.7 32.2 ± 1.2
FreD 38.4 ± 0.7 25.4 ± 1.7 31.9 ± 1.4 37.6 ± 2.0 19.7 ± 0.8 44.4 ± 1.0
DDiF **59.0 ± 0.4** **32.8 ± 0.8** 39.4 ± 0.8 **47.9 ± 0.6** **27.0 ± 0.6** **54.8 ± 1.1**
GSDD 53.4 ± 0.6 32.6 ± 1.1 **41.0 ± 1.2** 47.6 ± 0.8 25.8 ± 0.4 53.5 ± 2.1


**972**

**973**


**974**

**975**

**976**

**977**

**978**

**979**


**980**

**981**

**982**

**983**

**984**

**985**


**986**

**987**

**988**

**989**

**990**


**991**

**992**

**993**

**994**

**995**

**996**


**997**

**998**

**999**

**1000**

**1001**

**1002**


**1003**

**1004**

**1005**

**1006**

**1007**

**1008**


**1009**

**1010**

**1011**

**1012**

**1013**


**1014**

**1015**

**1016**

**1017**

**1018**

**1019**


**1020**

**1021**

**1022**

**1023**

**1024**

**1025**



Under review as a conference paper at ICLR 2026


real images, all Gaussian representations are first optimized to fit a single solid-color image. This
process reduces point sparsity and adjusts the initial point sizes, after which standard distillation
begins. As shown in Table 17, even without real-image initialization, our warmup-based strategy
achieves strong results and surpasses several methods that depend on real-image initialization.



B.5 RESULTS ON CROSS-RESOLUTION GENERALIZATION


Gaussian Splatting naturally supports rendering at arbitrary resolutions through supersampling, enabling us to distill data at a low resolution (e.g., 128) while still using it to train models at a higher
resolution (e.g., 256). This effectively reduces the computation when distilling datasets at large
resolutions. Following results and setup in DDiF Shin et al. (2025), we conduct cross-resolution
experiments as shown in Table 21.


C PARALLELIZATION STRATEGY FOR BATCHED RENDERING


To scale the rendering pipeline from a single image to an entire batch of images while maximizing
GPU throughput, we designed a global ID system coupled with corresponding data structure management. The core idea is to consolidate the rendering of multiple independent images into a single,
massive computational task.


C.1 CORE CHALLENGE AND THE GLOBAL ID SYSTEM


The parallelization strategy for batched rendering evolves the original single-image pipeline’s local
indexing system into a global, batch-aware framework. The original approach operated in a local
context where screen tiles were indexed from 0 to M-1 for a single image, and data structures
pertained only to that instance.


19



**More ablation study on individual components** We first provide an
ablation study on the ‘reg’ and ‘clip’ operations in the boundary constraints to supplement the boundary components reported in Table 6. As
shown in the results, both operations contribute to alleviating the Gaussian
escape problem. In addition to Table 6, we further perform an incremental ablation experiment starting from a baseline with no techniques, and
adding each technique individually. Table 13 shows that bf16 and opacity contribute the most to the final performance, whereas boundary and
antialiasing offer smaller gains. Similar trends can also be observed in
Table 6.


B.4 RESULTS ON IMAGENET-1K



Table 17: Initialization
Strategy Ablation Results


Init Acc.


IDC real 61.4
FreD real 66.8
NSD random 68.6
DDiF real 72.0
GSDD random 63.6
GSDD warmup 69.4
GSDD real 76.4



Enabled by the efficiency of recent dataset-distillation
frameworks, such as SRe2L Yin et al. (2023) and
RDED Sun et al. (2024), dataset parameterization at the
ImageNet-1K Deng et al. (2009) scale has become feasible for the first time. In this work, we build upon SRe2L
and evaluate our method under both IPC=1 and IPC=10
settings. In addition, we reproduce the IDC Kim et al.
(2022) baseline, where each image is spatially down_√_
scaled by a factor of _k_, resulting in _k_ -times more syn
thetic samples.



Table 18: Performance on ImageNet-1K.


IPC=1 IPC=10


SRe2L 0.9 21.3
SRe2L+IDC 13.4 41.5
SRe2L+GSDD 29.0 46.2


Table 19: Hyperparameters for each
method on ImageNet-1K.



IPC=1 IPC=10



For IDC, under IPC=1 we report the best-performing _k_
selected from _k ∈{_ 4 _,_ 20 _,_ 50 _}_, and under IPC=10 from SRe2L+IDC k=20 k=5

SRe2L+GSDD GPC=50 GPC=200

_k ∈{_ 4 _,_ 10 _,_ 20 _}_ . As shown in Table 18, our method improves the performance of SRe2L on ImageNet-1K, particularly in the IPC=1 setting. Although the gains diminish at IPC=10, our approach still maintains
a clear margin over IDC.



SRe2L+IDC k=20 k=5
SRe2L+GSDD GPC=50 GPC=200


**1026**

**1027**


**1028**

**1029**

**1030**

**1031**

**1032**

**1033**


**1034**

**1035**

**1036**

**1037**

**1038**

**1039**


**1040**

**1041**

**1042**

**1043**

**1044**


**1045**

**1046**

**1047**

**1048**

**1049**

**1050**


**1051**

**1052**

**1053**

**1054**

**1055**

**1056**


**1057**

**1058**

**1059**

**1060**

**1061**

**1062**


**1063**

**1064**

**1065**

**1066**

**1067**


**1068**

**1069**

**1070**

**1071**

**1072**

**1073**


**1074**

**1075**

**1076**

**1077**

**1078**

**1079**



Under review as a conference paper at ICLR 2026


To scale this process across a batch of B images, we introduced a Globally Unique ID (GUID)
system. This system creates a single, contiguous address space for all tiles across the entire batch.
A local tile j from image i is re-indexed into a global tile id using the linear transformation:


global ~~t~~ ile ~~i~~ d = _i × M_ + _j_ (12)


This re-indexing prompted adaptations to our key CUDA kernels. The intersection mapping kernel
was modified to calculate this global tile id for each generated intersection record, enabling the subsequent sorting operation to correctly group tasks on a global, batch-wide level. Correspondingly,
the rasterization kernel is now launched with a 1D grid of B×M thread blocks, where each block’s
index directly corresponds to a global tile id. Inside the kernel, this global ID is decomposed back
into its constituent image id and local tile id. This allows each thread block to precisely identify
which tile of which image it is responsible for, ensuring it writes the final pixel color to the correct
memory offset within the batch output tensor.


To support this parallelization strategy, we further employ specific data structures at the Python
(PyTorch) level to manage and transfer data efficiently.


C.2 1D FLATTENING AND CONTIGUOUS MEMORY LAYOUT


We avoid using Python lists or non-contiguous tensors to store the parameters of different images.
Instead, all Gaussian parameters (e.g., means, features, and Cholesky components) across the batch
are concatenated and stored in a single large contiguous PyTorch tensor. For a batch of _B_ images
with _N_ points each, the means tensor has a shape of ( _B × N,_ 2), rather than being represented
as a list of _B_ tensors of shape ( _N,_ 2). This design ensures memory contiguity and eliminates the
performance overhead caused by frequent concatenation.


By combining the global ID system at the CUDA kernel level with a contiguous memory layout at
the framework level, we have constructed an end-to-end rendering architecture that achieves high
throughput for large-scale batched rendering tasks.


C.3 BENCHMARK ACROSS DIFFERENT GPU ARCHITECTURES


To evaluate the stability of our CUDA-based rasterizer, we conduct systematic benchmarking on five
different GPU architectures. We use an image-fitting task in which Gaussian representations reconstruct 1000 images sampled from ImageNet (10 classes _×_ 100 images in total). Each experiment is
repeated three times, and we average all results. We test four storage budgets (21, 170, 682, 2730
Gaussians per image). All experiments used PyTorch 2.5.1 and CUDA _≤_ 12.0.


Table 20: Runtime, memory usage, and PSNR of our rasterizer benchmarked across five GPU architectures under varying storage budgets.

|GPU Model|Time (s)<br>21 170 682 2730|Memory (GB)<br>21 170 682 2730|PSNR<br>21 170 682 2730|
|---|---|---|---|
|Tesla-V100-SXM2-32GB<br>NVIDIA-A100-SXM4-40GB<br>NVIDIA-GeForce-RTX-3090<br>NVIDIA-GeForce-RTX-4090<br>NVIDIA-GeForce-RTX-5090|23.65<br>26.11<br>30.28<br>42.94<br>10.45<br>12.87<br>19.09<br>41.53<br>10.33<br>16.07<br>25.44<br>61.38<br>7.60<br>9.38<br>14.06<br>31.00<br>7.13<br>6.99<br>9.54<br>21.79|1.66<br>1.70<br>1.81<br>2.28<br>1.66<br>1.70<br>1.81<br>2.28<br>1.66<br>1.70<br>1.81<br>2.28<br>1.66<br>1.70<br>1.81<br>2.28<br>1.66<br>1.70<br>1.81<br>2.28|20.24<br>25.93<br>30.65<br>36.09<br>20.24<br>26.03<br>31.06<br>36.08<br>20.24<br>26.03<br>31.06<br>36.08<br>20.24<br>26.02<br>31.03<br>36.06<br>20.23<br>26.02<br>31.03<br>36.06|



As shown in Table 20. Different GPUs exhibit different runtimes due to heterogeneous compute
capabilities, and RTX5090 is the fastest, while RTX3090 is the slowest. Memory usage remained
identical across all platforms, and PSNR variation is small across architectures. Slight performance
drops on V100 are expected due to older hardware, but results on A100/3090/4090/5090 remain
highly consistent.


For IDC, SPEED, and FreD, we adopt their best-performing upsampling strategies. For DDiF and
GSDD, supersampling is applied, allowing lossless upscaling of the distilled images. GSDD maintains competitive performance in cross-resolution tasks. Although its relative performance drop is
larger than DDiF, which is likely due to GSDD’s inherent sparsity, reducing its ability to capture
fine-grained features when evaluated at higher resolutions.


20


**1080**

**1081**


**1082**

**1083**

**1084**

**1085**

**1086**

**1087**


**1088**

**1089**

**1090**

**1091**

**1092**

**1093**


**1094**

**1095**

**1096**

**1097**

**1098**


**1099**

**1100**

**1101**

**1102**

**1103**

**1104**


**1105**

**1106**

**1107**

**1108**

**1109**

**1110**


**1111**

**1112**

**1113**

**1114**

**1115**

**1116**


**1117**

**1118**

**1119**

**1120**

**1121**


**1122**

**1123**

**1124**

**1125**

**1126**

**1127**


**1128**

**1129**

**1130**

**1131**

**1132**

**1133**



Under review as a conference paper at ICLR 2026


Table 21: Test accuracies with different resolutions and networks. Images are first distilled with low
resolution(128) and test at higher resolution(256/512).


**test resolution** **test network** **method** **accuracy↑** **difference↓** **ratio↓**



Vanilla 31.2 20.2 0.39
IDC 55.0 6.4 0.10
SPEED 58.8 8.1 0.12
FreD 56.4 10.4 0.16
DDiF 66.3 **5.7** **0.08**
GSDD **66.4** 10.0 0.13


Vanilla 44.0 7.3 0.14
IDC 55.4 6.0 0.10
SPEED 62.6 4.3 0.06
FreD 61.8 5.0 0.07
DDiF 70.6 **1.4** **0.02**
GSDD **71.6** 4.8 0.06


Vanilla 27.4 24.0 0.47
IDC 39.5 21.9 0.36
SPEED 45.0 21.9 0.33
FreD 42.9 23.9 0.36
DDiF 58.7 **13.3** **0.18**
GSDD **59.4** 17.0 0.22


Vanilla 41.2 10.1 0.20
IDC 51.5 9.9 0.16
SPEED 60.1 6.8 0.10
FreD 56.3 10.5 0.16
DDiF **69.0** **3.0** **0.04**
GSDD 67.8 8.6 0.11


21



256


512



ConvNetD5


ConvNetD6


ConvNetD5


ConvNetD6


**1134**

**1135**


**1136**

**1137**

**1138**

**1139**

**1140**

**1141**


**1142**

**1143**

**1144**

**1145**

**1146**

**1147**


**1148**

**1149**

**1150**

**1151**

**1152**


**1153**

**1154**

**1155**

**1156**

**1157**

**1158**


**1159**

**1160**

**1161**

**1162**

**1163**

**1164**


**1165**

**1166**

**1167**

**1168**

**1169**

**1170**


**1171**

**1172**

**1173**

**1174**

**1175**


**1176**

**1177**

**1178**

**1179**

**1180**

**1181**


**1182**

**1183**

**1184**

**1185**

**1186**

**1187**



Under review as a conference paper at ICLR 2026


(a) (b)


Figure 7: Comparison between aliasing and anti-aliasing. (a) shows the schematic illustration, while
(b) demonstrates the effect on CIFAR-10 dataset visualization. The anti-aliasing strategy leads to
smoother and clearer patterns.


D VISUALIZATION OF ANTIALIAS AND GAUSSIAN ESCAPE


(a) CIFAR-10 (b) Imagenette


Figure 8: Visualization of Gaussian point distributions without boundary constraints. We plot the
center coordinates of all Gaussian points after training and observe clear escaping phenomena on
both CIFAR-10 and Imagenette, which reduces the representational efficiency of Gaussian points.


Figure 7 visualizes the test images without and with anti-aliasing, along with the actual renderings
on CIFAR-10. It is evident that when the Gaussian points are relatively elongated, the rendering
quality degrades, resulting in pronounced aliasing artifacts and discontinuous stripes, which in turn
adversely affect the performance of the distilled dataset.


In Figure 8, we visualize the escaping phenomenon of Gaussian points after training when no boundary constraints are applied. We plot the distributions of all Gaussian point centers and observe that,
on both CIFAR-10 and Imagenette datasets, Gaussian points tend to escape beyond the valid range,
which in turn undermines their representational efficiency.


22


**1188**

**1189**


**1190**

**1191**

**1192**

**1193**

**1194**

**1195**


**1196**

**1197**

**1198**

**1199**

**1200**

**1201**


**1202**

**1203**

**1204**

**1205**

**1206**


**1207**

**1208**

**1209**

**1210**

**1211**

**1212**


**1213**

**1214**

**1215**

**1216**

**1217**

**1218**


**1219**

**1220**

**1221**

**1222**

**1223**

**1224**


**1225**

**1226**

**1227**

**1228**

**1229**


**1230**

**1231**

**1232**

**1233**

**1234**

**1235**


**1236**

**1237**

**1238**

**1239**

**1240**

**1241**



Under review as a conference paper at ICLR 2026


E VISUALIZATION OF GSDD


Figure 9: Visualization of the Gaussian initialization process. The first row shows the actual renderings of Gaussian points at different iterations, while the second row represents each Gaussian as an
ellipse encoding its position, shape, and color. During optimization, Gaussian points continuously
adjust their positions, shapes, and colors to fit the target image.


We visualize the process of Gaussian initialization in Figure 9. The first row illustrates the actual
rendering results of Gaussian points, while the second row depicts the Gaussians as ellipses that
encode their positions, shapes, and colors. It can be observed that, during optimization, Gaussian
points continuously move, reshape, and adjust their colors to fit the target image.


Figure 10: Visualization of Distilled Images on CIFAR-10


23


**1242**

**1243**


**1244**

**1245**

**1246**

**1247**

**1248**

**1249**


**1250**

**1251**

**1252**

**1253**

**1254**

**1255**


**1256**

**1257**

**1258**

**1259**

**1260**


**1261**

**1262**

**1263**

**1264**

**1265**

**1266**


**1267**

**1268**

**1269**

**1270**

**1271**

**1272**


**1273**

**1274**

**1275**

**1276**

**1277**

**1278**


**1279**

**1280**

**1281**

**1282**

**1283**


**1284**

**1285**

**1286**

**1287**

**1288**

**1289**


**1290**

**1291**

**1292**

**1293**

**1294**

**1295**



Under review as a conference paper at ICLR 2026


Figure 11: Visualization of Distilled Images on CIFAR-100


Figure 12: Visualization of Distilled Images on ImageNette


24


