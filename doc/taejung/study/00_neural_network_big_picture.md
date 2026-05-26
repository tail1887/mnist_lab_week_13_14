# 신경망 전체 구조 학습 문서

## 1. 이 문서의 목적

이 문서는 `ReLU`, `Softmax`, `Affine`, `BatchNorm`, `Dropout`, `Loss`, `Optimizer`를 하나씩 보기 전에, 전체 신경망이 어떤 배경에서 만들어졌고 왜 지금과 같은 구조를 갖는지 설명한다.

이번 MNIST 과제의 모델은 다음과 같다.

```text
입력 784
-> Affine(512)
-> BatchNorm
-> ReLU
-> Dropout
-> Affine(256)
-> BatchNorm
-> ReLU
-> Dropout
-> Affine(10)
-> Softmax
-> Cross Entropy Loss
```

겉으로 보면 여러 부품을 순서대로 붙인 것처럼 보인다. 하지만 각 부품은 신경망 학습의 오래된 문제를 해결하기 위해 생겼다.

```text
Affine: 입력 특징을 조합한다.
Activation: 선형 모델의 한계를 넘는다.
Softmax + Loss: 분류 문제를 확률과 최적화 문제로 바꾼다.
Backward: 각 파라미터가 loss에 미친 영향을 계산한다.
Optimizer: loss가 줄어드는 방향으로 파라미터를 바꾼다.
BatchNorm: 학습 중 값의 분포를 안정화한다.
Dropout: 과적합을 줄인다.
```

## 2. 신경망은 무엇을 하려는 모델인가

신경망의 목표는 입력 `x`를 받아 정답 `y`에 가까운 출력을 내는 함수 `f`를 학습하는 것이다.

```text
입력 x -> f(x; parameters) -> 예측값
```

여기서 `parameters`는 모델이 학습하는 값이다. 이번 과제에서는 `W`, `b`, `gamma`, `beta` 같은 값들이 파라미터다.

MNIST에서는 입력이 28x28 이미지를 펼친 784차원 벡터다.

```text
x: (784,)
```

출력은 숫자 0부터 9까지 10개 클래스 중 하나다.

```text
y: 0, 1, 2, ..., 9
```

따라서 신경망은 다음 함수를 학습하려고 한다.

```text
f: R^784 -> R^10
```

`R^10`의 각 값은 처음에는 단순한 점수이며, 마지막에 `Softmax`를 통과해 클래스별 확률이 된다.

## 3. 퍼셉트론에서 시작된 구조

신경망의 기본 아이디어는 퍼셉트론에서 출발한다.

퍼셉트론은 여러 입력에 가중치를 곱해 더한 뒤, 임계값을 넘으면 1, 아니면 0을 출력하는 모델이다.

```text
z = w_1x_1 + w_2x_2 + ... + w_nx_n + b

z > 0 이면 1
z <= 0 이면 0
```

이를 벡터로 쓰면 다음과 같다.

```text
z = xW + b
```

여기서 이미 현재 모델의 `Affine` 구조가 나온다.

```text
Affine = weighted sum + bias
```

퍼셉트론은 입력 공간을 하나의 직선이나 평면으로 나누는 모델이다. 2차원에서는 직선, 3차원에서는 평면, 고차원에서는 초평면으로 나눈다.

```text
w_1x_1 + w_2x_2 + b = 0
```

이 선을 기준으로 한쪽은 클래스 0, 다른 쪽은 클래스 1로 분류한다.

## 4. 단층 퍼셉트론의 한계

단층 퍼셉트론은 선형 분리 가능한 문제만 풀 수 있다.

대표적인 한계가 XOR 문제다.

```text
x1 x2 | y
0  0  | 0
0  1  | 1
1  0  | 1
1  1  | 0
```

이 네 점은 하나의 직선으로 0과 1을 나눌 수 없다. 즉, 단순한 선형 모델은 표현력이 부족하다.

MNIST도 마찬가지다. 숫자 2와 3, 4와 9 같은 이미지는 픽셀 공간에서 단순한 하나의 선형 경계만으로 깔끔하게 나누기 어렵다.

그래서 여러 개의 선형 변환을 쌓고, 그 사이에 비선형 함수를 넣는 구조가 필요해졌다.

```text
Affine -> Activation -> Affine -> Activation -> Affine
```

이것이 다층 퍼셉트론, 즉 `MLP(Multi-Layer Perceptron)`다.

## 5. 왜 Affine만 여러 번 쌓으면 안 되는가

만약 비선형 활성화 없이 `Affine`만 여러 번 쌓는다고 하자.

```text
h = xW1 + b1
y = hW2 + b2
```

두 식을 합치면:

```text
y = (xW1 + b1)W2 + b2
  = x(W1W2) + b1W2 + b2
```

이는 다시 하나의 affine 변환이다.

```text
y = xW_new + b_new
```

즉, `Affine`을 아무리 많이 쌓아도 중간에 비선형성이 없으면 결국 하나의 선형 모델과 같은 표현력만 가진다.

그래서 `ReLU` 같은 활성화 함수가 필요하다.

```text
Affine -> ReLU -> Affine -> ReLU -> Affine
```

비선형 함수가 들어가면 층을 쌓을수록 입력 공간을 더 복잡하게 접고 나눌 수 있다.

## 6. 깊은 신경망의 직관: 표현 학습

신경망의 중요한 특징은 사람이 직접 특징을 설계하지 않아도 중간층이 특징을 학습한다는 점이다.

MNIST에서 입력은 픽셀이다.

```text
784개 픽셀
```

하지만 사람은 숫자를 픽셀 하나하나로만 보지 않는다. 획, 곡선, 닫힌 부분, 기울기 같은 특징을 보고 숫자를 판단한다.

신경망의 은닉층은 이런 중간 표현을 스스로 만든다.

```text
입력 픽셀
-> 낮은 수준 특징: 선, 획, 방향
-> 높은 수준 특징: 숫자 모양의 부분 패턴
-> 출력 클래스: 0~9
```

물론 이번 과제의 MLP는 CNN처럼 공간 구조를 직접 활용하지는 않는다. 이미지를 784차원 벡터로 펼치기 때문에 위치 관계를 완벽히 보존하지는 못한다. 그래도 충분한 은닉층과 비선형성을 통해 픽셀 조합의 패턴을 학습할 수 있다.

## 7. 현재 모델 구조가 의미하는 것

현재 모델은 다음 차원 흐름을 가진다.

```text
(batch_size, 784)
-> (batch_size, 512)
-> (batch_size, 256)
-> (batch_size, 10)
```

각 단계의 의미는 다음과 같다.

```text
784:
  28x28 이미지를 펼친 원본 픽셀

512:
  원본 픽셀을 조합해 첫 번째 은닉 표현 생성

256:
  첫 번째 은닉 표현을 다시 조합해 더 압축된 표현 생성

10:
  각 숫자 클래스에 대한 점수
```

출력층의 10개 값은 아직 확률이 아니다. 이 값은 `logit` 또는 클래스 점수라고 부른다.

```text
logits = [score_0, score_1, ..., score_9]
```

이 점수는 `Softmax`를 통과한 뒤 확률이 된다.

## 8. Forward Propagation

Forward propagation은 입력이 모델을 통과해 예측값이 되는 과정이다.

```text
x
-> Affine
-> BatchNorm
-> ReLU
-> Dropout
-> Affine
-> BatchNorm
-> ReLU
-> Dropout
-> Affine
-> Softmax
-> y_pred
```

수학적으로는 함수 합성이다.

```text
y_pred = f(x)
```

조금 더 풀어 쓰면:

```text
h1 = Affine1(x)
h1_norm = BatchNorm1(h1)
a1 = ReLU(h1_norm)
d1 = Dropout(a1)

h2 = Affine2(d1)
h2_norm = BatchNorm2(h2)
a2 = ReLU(h2_norm)
d2 = Dropout(a2)

logits = Affine3(d2)
y_pred = Softmax(logits)
```

각 layer는 backward에서 쓸 값을 forward 중에 저장한다.

```text
Affine: 입력 x 저장
ReLU: mask 저장
BatchNorm: mean, var, x_norm 저장
Dropout: mask 저장
```

이 저장값들이 있어야 나중에 gradient를 계산할 수 있다.

## 9. Loss: 예측을 숫자 하나로 평가하기

모델 출력은 10개 확률이다. 하지만 학습하려면 “얼마나 틀렸는지”를 하나의 숫자로 만들어야 한다.

이 숫자가 loss다.

MNIST 같은 다중 클래스 분류에서는 `Softmax + Cross Entropy`를 많이 사용한다.

```text
Softmax:
  logit을 확률로 바꾼다.

Cross Entropy:
  정답 클래스 확률이 낮을수록 큰 벌점을 준다.
```

정답 클래스 확률을 `p`라고 하면:

```text
loss = -log(p)
```

이 식은 최대우도 추정과 연결된다. 모델이 정답에 높은 확률을 주도록 만들고 싶다면 `p`를 최대화해야 한다.

```text
maximize p
```

최적화에서는 보통 최소화 문제를 풀기 때문에 음의 로그를 취한다.

```text
minimize -log(p)
```

여러 샘플에 대해서는 평균을 낸다.

```text
loss = -(1/N) * sum_n log(p_n)
```

## 10. Backward Propagation

Backward propagation은 loss를 줄이기 위해 각 파라미터를 어느 방향으로 바꿔야 하는지 계산하는 과정이다.

핵심은 체인 룰이다.

```text
z = f(y)
y = g(x)

dz/dx = dz/dy * dy/dx
```

신경망은 여러 함수가 이어진 구조다.

```text
x -> layer1 -> layer2 -> layer3 -> loss
```

따라서 loss가 앞쪽 파라미터에 미친 영향은 뒤쪽부터 차례대로 미분을 곱해 계산한다.

```text
dLoss/dW1
= dLoss/dout
* dout/dlayer3
* dlayer3/dlayer2
* dlayer2/dlayer1
* dlayer1/dW1
```

이것이 backward가 forward의 반대 순서로 진행되는 이유다.

```text
forward:
Affine1 -> ReLU1 -> Affine2 -> ReLU2 -> Affine3

backward:
Affine3 -> ReLU2 -> Affine2 -> ReLU1 -> Affine1
```

각 layer는 자기 입력에 대한 gradient를 계산해 이전 layer로 넘긴다. 파라미터가 있는 layer는 파라미터 gradient도 저장한다.

```text
Affine:
  dW, db, dx

BatchNorm:
  dgamma, dbeta, dx

ReLU:
  dx

Dropout:
  dx
```

## 11. Optimizer: 계산한 gradient로 파라미터 바꾸기

Backward는 gradient를 계산할 뿐 파라미터를 직접 바꾸지는 않는다.

파라미터 업데이트는 optimizer가 담당한다.

가장 기본은 SGD다.

```text
W = W - lr * dW
```

이는 loss를 1차 테일러 전개로 근사했을 때, gradient 반대 방향으로 조금 이동하면 loss가 줄어든다는 원리에서 나온다.

Adam은 여기에 두 가지 기억을 추가한다.

```text
m: gradient의 이동평균
v: gradient 제곱의 이동평균
```

그래서 Adam은 단순히 현재 gradient만 보는 것이 아니라, 최근 gradient의 방향과 크기를 함께 고려한다.

```text
SGD:
  지금 gradient만 사용

Adam:
  지금 gradient + 과거 방향 평균 + 과거 크기 평균
```

## 12. BatchNorm이 들어가는 이유

학습 중 각 층의 파라미터는 계속 바뀐다. 그러면 다음 층이 받는 입력 분포도 계속 바뀐다.

```text
앞 층 업데이트
-> 앞 층 출력 분포 변경
-> 뒤 층 입력 분포 변경
```

BatchNorm은 mini-batch 단위로 각 feature의 평균과 분산을 맞춰 이 변화를 줄인다.

```text
x_norm = (x - mean) / sqrt(var + eps)
out = gamma * x_norm + beta
```

여기서 `gamma`, `beta`를 두는 이유는 정규화만 하면 표현력이 제한될 수 있기 때문이다. 모델이 필요하면 scale과 shift를 다시 학습할 수 있게 한다.

현재 구조에서는 `Affine` 뒤, `ReLU` 앞에 BatchNorm이 들어간다.

```text
Affine -> BatchNorm -> ReLU
```

`Affine`이 만든 값을 먼저 안정화한 뒤, ReLU로 비선형성을 적용하는 흐름이다.

## 13. Dropout이 들어가는 이유

신경망은 파라미터가 많기 때문에 학습 데이터에 과하게 맞춰질 수 있다. 이를 과적합이라고 한다.

Dropout은 학습 중 일부 뉴런을 무작위로 끈다.

```text
y = x * mask
```

이렇게 하면 모델이 특정 뉴런 하나나 특정 뉴런 조합에 과하게 의존하기 어렵다.

```text
오늘은 이 뉴런이 꺼질 수도 있음
-> 다른 뉴런들도 쓸모 있는 특징을 배워야 함
```

Dropout은 여러 작은 모델을 랜덤하게 학습하는 ensemble과 비슷한 효과를 낸다.

현재 구조에서는 `ReLU` 뒤에 Dropout이 들어간다.

```text
Affine -> BatchNorm -> ReLU -> Dropout
```

활성화된 은닉 표현 중 일부를 꺼서 과적합을 줄이는 역할이다.

## 14. 학습 모드와 추론 모드가 다른 이유

일부 layer는 학습할 때와 추론할 때 다르게 동작한다.

```text
BatchNorm:
  train=True  -> 현재 batch의 mean/var 사용
  train=False -> running mean/var 사용

Dropout:
  train=True  -> 랜덤 mask 적용
  train=False -> 랜덤 mask 없이 scale 적용
```

학습 중에는 gradient를 계산하고 모델을 튼튼하게 만들기 위해 batch 통계와 랜덤성이 들어간다.

하지만 추론 중에는 같은 입력에 대해 안정적인 출력이 나와야 한다. 그래서 Dropout은 랜덤하게 끄지 않고, BatchNorm은 누적된 running 통계를 사용한다.

## 15. 왜 He 초기화를 쓰는가

신경망은 처음에 가중치를 랜덤하게 초기화한다. 모든 가중치를 0으로 두면 모든 뉴런이 같은 출력을 내고 같은 gradient를 받아 똑같이 학습된다. 그러면 여러 뉴런을 둔 의미가 사라진다.

```text
모든 W가 같음
-> 모든 뉴런이 같은 계산
-> 모든 gradient도 같음
-> 계속 같은 뉴런으로 남음
```

그래서 랜덤 초기화가 필요하다.

하지만 너무 큰 값으로 초기화하면 forward 값과 gradient가 폭발할 수 있고, 너무 작은 값으로 초기화하면 값과 gradient가 사라질 수 있다.

현재 코드는 ReLU에 맞는 He 초기화를 사용한다.

```text
W ~ N(0, sqrt(2 / fan_in))
```

ReLU는 음수 절반을 0으로 만들 가능성이 있으므로, 이를 고려해 분산을 조금 크게 잡는다. 이렇게 하면 층을 지날 때 값의 스케일이 너무 빨리 작아지거나 커지는 것을 줄일 수 있다.

## 16. 전체 학습 루프

이번 과제의 학습 흐름은 다음과 같다.

```text
for epoch:
    shuffle training data

    for mini-batch:
        1. y_pred = model.forward(x_batch, train=True)
        2. loss = cross_entropy_loss(y_pred, y_batch)
        3. dout = (y_pred - one_hot(y_batch)) / batch_size
        4. model.backward(dout)
        5. optimizer.update(model.params, model.grads)
```

이 루프는 신경망 학습의 핵심을 모두 담고 있다.

```text
예측한다.
틀린 정도를 계산한다.
각 파라미터 책임을 계산한다.
파라미터를 조금 고친다.
반복한다.
```

학습은 한 번에 정답 공식을 찾는 과정이 아니라, loss가 줄어드는 방향으로 파라미터를 계속 조금씩 조정하는 과정이다.

## 17. 왜 이런 순서인가

현재 은닉층 순서는 다음과 같다.

```text
Affine -> BatchNorm -> ReLU -> Dropout
```

각 단계의 이유는 다음과 같다.

```text
Affine:
  입력 특징을 가중합으로 조합한다.

BatchNorm:
  Affine 출력의 분포를 안정화한다.

ReLU:
  비선형성을 넣어 복잡한 패턴을 표현한다.

Dropout:
  활성화된 표현 일부를 꺼서 과적합을 줄인다.
```

출력층은 다음과 같다.

```text
Affine(10) -> Softmax
```

마지막 `Affine`은 10개 클래스 점수를 만들고, `Softmax`는 이를 확률로 바꾼다.

손실은 `Cross Entropy`를 사용한다.

```text
Softmax + Cross Entropy
```

이 조합은 다중 클래스 분류에서 자연스럽고, 미분 결과도 간단하다.

```text
dout = y_pred - one_hot(y_true)
```

## 18. 큰 관점에서 보는 신경망

신경망은 단순히 수식을 많이 붙인 것이 아니다.

큰 관점에서 보면 다음 문제를 푸는 구조다.

```text
1. 입력 공간을 더 분류하기 쉬운 표현 공간으로 바꾼다.
2. 그 표현 공간에서 클래스별 점수를 만든다.
3. 점수를 확률로 바꾼다.
4. 정답과의 차이를 loss로 만든다.
5. 체인 룰로 각 파라미터의 책임을 계산한다.
6. optimizer로 파라미터를 갱신한다.
```

즉, 신경망 학습은 다음 두 과정의 반복이다.

```text
Forward:
  현재 파라미터로 얼마나 잘 예측하는지 확인

Backward + Update:
  더 잘 예측하도록 파라미터를 수정
```

처음에는 랜덤한 함수였던 모델이, 이 반복을 통해 MNIST 숫자를 구분하는 함수로 바뀐다.

## 19. 이 과제에서 특히 중요한 연결

각 구성 요소를 따로 외우는 것보다, 다음 연결을 이해하는 것이 중요하다.

```text
Affine forward에서 저장한 x
-> Affine backward의 dW 계산에 필요

ReLU forward에서 저장한 mask
-> ReLU backward의 gradient 차단에 필요

Softmax 출력 y_pred
-> Cross Entropy loss 계산에 필요
-> y_pred - one_hot 형태의 gradient 생성에 필요

BatchNorm forward의 x_norm, std
-> BatchNorm backward의 dgamma, dx 계산에 필요

Dropout forward의 mask
-> Dropout backward의 gradient 차단에 필요

model.grads
-> optimizer.update에서 params 갱신에 필요
```

즉, forward는 단순히 예측만 만드는 과정이 아니다. backward에 필요한 증거를 저장하는 과정이기도 하다.

## 20. 다음에 읽으면 좋은 순서

전체 그림을 이해한 뒤에는 다음 순서로 개별 문서를 읽으면 연결이 자연스럽다.

```text
00_neural_network_big_picture.md
-> 03_affine.md
-> 01_relu.md
-> 02_softmax.md
-> 04_cross_entropy_loss.md
-> 05_sgd.md
-> 06_adam.md
-> 07_neural_network.md
-> 08_batchnorm.md
-> 09_dropout.md
```

처음에는 forward 흐름을 먼저 이해하고, 그 다음 backward와 optimizer를 보면 좋다. 마지막으로 BatchNorm과 Dropout을 보면 “기본 신경망이 돌아간 뒤, 왜 안정화와 일반화 기법이 추가되는지”가 더 잘 보인다.
