# markov_irl
文章のマルコフ連鎖から逆強化学習で報酬を推定したい

## 何がしたいか
単語の連鎖から文章を生成するのがマルコフ連鎖によるテキスト生成だが、時として「文法は合っているけど奇妙な文」ができてしまう。
強化学習におけるMDPにおいて、S＝単語、A＝ある単語への連鎖行動、T＝単語間遷移確率、R＝遷移時報酬、とみなせば、文章生成も強化学習で行うことができないか。
ということは、逆強化学習で、「この単語を重視すべき」という報酬が計算できるはず

## method
Abbeel(2004)の逆強化学習(projection method)を使う。強化学習は方策反復。

## Usage
> python Encoder.py

で、list.txtを用いて単語列のIDを作る。

そして、

> python markov_irl.py

で、逆強化学習による単語ごとの報酬が計算される。
