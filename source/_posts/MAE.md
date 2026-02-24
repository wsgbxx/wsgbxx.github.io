---
title: MAKE MAE GREAT AGAIN
date: 2026-02-11 00:15:56
categories:
  - 项目
  - 生成模型之旅
tags:
  - 生成模型    
---
实际上我没有看过MAE的paper，本次是对pixio的复现结果，当然基本架构都差不多，pixio的效果也更好。
pixio repo：[https://github.com/facebookresearch/pixio](https://github.com/facebookresearch/pixio)

## 模型性能

在monodepth和semantic任务上测试结果超过或与dinov3持平，但是结构更加简单，只是对MAE进行了简单但暴力的改进。

## abstract
- 更大的mask
- 更多的cls token
- 更大的数据集webdatasets（闭源）

## experiment 
作者没有开源decoder权重，我冻结vith16的encoder，在imagenet1k上训练了decoder。
分辨率为224*224，由于patchsize为16，所以图片会被分成14*14个patch，有点马赛克的效果。

## results 
虽然比较简陋但还是能体现出来，模型真的有学到语义信息

### imagenet val数据集测试效果
{% asset_img "MAKE MAE GREAT AGAIN_3_老师好我叫苏同学_来自小红书网页版.jpg" %}
{% asset_img "MAKE MAE GREAT AGAIN_5_老师好我叫苏同学_来自小红书网页版.jpg" %}

说实话我觉得已经超越了普通人类的水平了，甚至有些过拟合的迹象
于是使用了自己的手机图片进行了测试
### 手机图片测试
{% asset_img "MAKE MAE GREAT AGAIN_2_老师好我叫苏同学_来自小红书网页版.jpg" %}
{% asset_img "MAKE MAE GREAT AGAIN_4_老师好我叫苏同学_来自小红书网页版.jpg" %}  

怎么说呢，嗯~~，确实强得很可怕，在语义方面比我手机里自带的AI修图要强很多

## future work
我有一个大胆的想法：让pixio先推理做high-level处理，然后用AI修图工具做low-level处理。
不过自己试了一下，效果一般哈哈

## Anyway，只要数据集足够大，过拟合即智能！