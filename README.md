# Simple DCGAN and ProgressiveGAN Implementation and GAN Testing GUI

### 3-2 模型及訓練參數
#### 3-2-1 Generator設計
![]( https://i.imgur.com/o4lAnBg.png )
>  1. FC : 全連接層，輸入8192，輸出4096。啟動函數LeakyReLU(alpha=0.01)。
>  2. Reshape : 將輸入4096改為3維3*64*64，已便使用捲積計算。
>  3. CONV Transpose : 反捲積層，輸入64*8*8，128個4*4 kernel，stride為2, padding為1，輸出為128*16*16。啟動函數LeakyReLU(alpha=0.02)。
>  4. CONV Transpose : 反捲積層，輸入128*16*16，256個4*4 kernel，stride為2, padding為1，輸出為256*32*32。啟動函數LeakyReLU(alpha=0.02)。
>  5. CONV Transpose : 反捲積層，輸入256*32*32，512個4*4 kernel，stride為2, padding為1，輸出為512*64*64。啟動函數LeakyReLU(alpha=0.02)。
>  6. CONV Transpose : 捲積層，輸入512*64*64，3個5*5 kernel，stride為1, padding為2，輸出為3*64*64。啟動函數使用Sigmoid。
#### 3-2-2 Discriminator設計
![]( https://i.imgur.com/GUdvRsG.png )
>  1. CONV : 反捲積層，輸入3*64*64，128個4*4 kernel，stride為2, padding為1，輸出為64*32*32啟動函數LeakyReLU(alpha=0.02)。
>  2. CONV : 反捲積層，輸入64*32*32，256個4*4 kernel，stride為2, padding為1，輸出為128*16*16。啟動函數LeakyReLU(alpha=0.02)。
>  3. CONV : 反捲積層，輸入128*16*16，512個4*4 kernel，stride為2, padding為1，輸出為256*8*8。啟動函數LeakyReLU(alpha=0.02)。
>  4. CONV : 捲積層，輸入256*8*8，3個4*4 kernel，stride為2, padding為1，輸出為512*4*4。啟動函數使用Sigmoid。
>  5. Reshape : 將輸入512*4*4改為1維8192。
>  6. FC : 全連接層，輸入8192，輸出1。啟動函數Sigmoid。Dropout設0.2。
#### 3-3 模型訓練結果
![]( https://i.imgur.com/uXNPxDy.png )

#### 3-4 GUI 程式
![]( https://i.imgur.com/BcqZ05V.png )

### 訓練結果
DCGAN:
(https://imgur.com/kZdczmj)

ProgressiveGAN:
(https://imgur.com/RJEhTRG)



##### 參考資料
>[1] Alec Radford, Luke Metz, Soumith Chintala. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. abs/1511.06434, 2016.
>
>[2] another anime face dataset, 2021 .URL https://www.kaggle.com/datasets/scribbless/another-anime-face-dataset
>
>[3] Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen. Progressive Growing of GANs for Improved Quality, Stability, and Variation. abs/1710.10196, 2018.
