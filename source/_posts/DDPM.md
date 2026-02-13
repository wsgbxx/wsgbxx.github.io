---
title: DDPM
date: 2026-02-11 00:16:05
categories:
  - 项目
  - 生成模型之旅
tags:
  - 生成模型
---
DDPM是diffusion的开山之作，diffusion也是当下主流的生图基础结构。在cifar 10上训练了200个epoch就很获得了很好的效果。

{% asset_img "U-net ，VAE，DDPM初体验_3_老师好我叫苏同学_来自小红书网页版.jpg" %}
图三为第5个epoch，可以看到只有一点色彩和纹理

{% asset_img "U-net ，VAE，DDPM初体验_4_老师好我叫苏同学_来自小红书网页版.jpg" %}
图四为25个epoch，更加有区分度

{% asset_img "U-net ，VAE，DDPM初体验_5_老师好我叫苏同学_来自小红书网页版.jpg" %}
图五为最终结果，可以说已经有了很明显的动物和汽车形态，只是受限于32×32的分辨率已经没办法再进一步了。

{% asset_img "U-net ，VAE，DDPM初体验_6_老师好我叫苏同学_来自小红书网页版.jpg" %}
最后用Unet简单训了一个32到96的超分网络（仅能看到分辨率的提升，相比直接插值没有什么明显的超分效果）