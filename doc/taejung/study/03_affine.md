# Affine 학습 문서

## 1. 핵심 역할

`Affine`은 완전연결층에서 사용하는 선형 변환이다.

```text
y = xW + b
```

MNIST 모델에서는 784차원 입력을 은닉층 차원으로 바꾸고, 마지막에는 10개 클래스 점수로 바꾼다.

```text
784 -> 512 -> 256 -> 10
```

여기서 `W`는 가중치, `b`는 편향이다. `Affine` 층은 입력 특징을 다음 층에서 쓰기 좋은 새로운 표현으로 바꾸는 역할을 한다.

## 배경: 왜 Affine 변환인가

신경망의 기본 계산은 입력들의 가중합이다.

```text
y = w_1x_1 + w_2x_2 + ... + w_nx_n
```

이 식은 “각 입력 특징을 얼마나 중요하게 볼지”를 가중치 `w`로 정하는 구조다. MNIST에서는 784개 픽셀 각각이 입력 특징이므로, 각 픽셀에 서로 다른 가중치를 곱해 다음 뉴런의 값을 만든다.

여기에 편향 `b`를 더하면 다음과 같다.

```text
y = w_1x_1 + w_2x_2 + ... + w_nx_n + b
```

편향은 입력이 모두 0이어도 뉴런이 가질 수 있는 기본값을 만든다. 기하학적으로 보면 결정 경계를 원점에 묶어두지 않고 이동시킬 수 있게 해준다.

엄밀히 말하면 `xW`는 선형 변환이고, `xW + b`는 affine 변환이다. 선형 변환은 원점을 반드시 원점으로 보내지만, affine 변환은 편향 덕분에 평행 이동이 가능하다.

```text
linear: y = xW
affine: y = xW + b
```

신경망에서는 이 affine 변환 뒤에 `ReLU` 같은 비선형 함수를 붙여 복잡한 함수를 표현한다.

## 행렬식으로 묶는 이유

샘플 하나만 보면 한 뉴런의 계산은 가중합이다.

```text
y_j = sum_i x_i W_ij + b_j
```

출력 뉴런이 여러 개이면 같은 입력 `x`에 대해 여러 개의 가중치 묶음이 필요하다. 이를 행렬로 모으면 다음처럼 한 번에 계산할 수 있다.

```text
y = xW + b
```

batch까지 포함하면 여러 샘플을 동시에 계산한다.

```text
X: (N, D)
W: (D, H)
b: (H,)
Y: (N, H)
```

여기서 `N`은 batch 크기, `D`는 입력 차원, `H`는 출력 차원이다.

## 2. Forward 흐름

현재 구현 파일은 `src/layers.py`다.

```python
self.x = x
out = x @ self.W + self.b
```

흐름은 다음과 같다.

1. 입력 `x`를 `self.x`에 저장한다.
2. 행렬곱 `x @ W`를 계산한다.
3. 편향 `b`를 더한다.
4. 결과를 다음 층으로 넘긴다.

shape 예시는 다음과 같다.

```text
x:   (batch_size, input_dim)
W:   (input_dim, output_dim)
b:   (output_dim,)
out: (batch_size, output_dim)
```

예를 들어 첫 번째 층에서는 다음과 같다.

```text
x:   (batch_size, 784)
W1:  (784, 512)
b1:  (512,)
out: (batch_size, 512)
```

## 3. Backward 흐름

현재 구현은 다음과 같다.

```python
self.dW = self.x.T @ dout
self.db = np.sum(dout, axis=0)
dx = dout @ self.W.T
```

`dout`은 다음 층에서 넘어온 gradient이며 shape는 `(batch_size, output_dim)`이다.

각 gradient의 의미는 다음과 같다.

- `dW`: 가중치 `W`가 loss에 미치는 영향
- `db`: 편향 `b`가 loss에 미치는 영향
- `dx`: 입력 `x`가 loss에 미치는 영향

shape 흐름은 다음과 같다.

```text
dout: (batch_size, output_dim)

dW = x.T @ dout
   = (input_dim, batch_size) @ (batch_size, output_dim)
   = (input_dim, output_dim)

db = sum(dout, axis=0)
   = (output_dim,)

dx = dout @ W.T
   = (batch_size, output_dim) @ (output_dim, input_dim)
   = (batch_size, input_dim)
```

`dW`와 `db`는 optimizer가 파라미터를 업데이트할 때 사용하고, `dx`는 이전 층으로 넘겨 역전파를 계속 진행하는 데 사용한다.

## Backward 공식 유도

샘플 하나와 출력 하나만 먼저 보면 다음과 같다.

```text
y_j = sum_i x_i W_ij + b_j
```

다음 층에서 `dL/dy_j`가 넘어왔다고 하자. 이를 `dout_j`라고 부른다.

가중치 `W_ij`에 대한 미분은 다음과 같다.

```text
dy_j/dW_ij = x_i
dL/dW_ij = dL/dy_j * dy_j/dW_ij
         = dout_j * x_i
```

batch 전체에 대해서는 모든 샘플의 기여를 더한다.

```text
dW_ij = sum_n x_ni * dout_nj
```

이 식을 행렬로 쓰면 다음이 된다.

```text
dW = X.T @ dout
```

편향은 각 출력에 그냥 더해진다.

```text
dy_j/db_j = 1
dL/db_j = sum_n dout_nj
```

그래서 코드에서는 batch 방향으로 합산한다.

```text
db = sum(dout, axis=0)
```

입력 `x_i`에 대한 미분은 모든 출력 뉴런으로부터 영향을 받는다.

```text
dL/dx_i = sum_j dL/dy_j * dy_j/dx_i
        = sum_j dout_j * W_ij
```

이를 행렬로 쓰면 다음과 같다.

```text
dx = dout @ W.T
```

## 4. 구현 포인트

Forward에서 입력 `x`를 저장해야 backward에서 `dW`를 계산할 수 있다.

```python
self.x = x
```

`dW = x.T @ dout`이므로, forward 입력이 없으면 가중치가 loss에 어떤 영향을 줬는지 계산할 수 없다.

편향 `b`는 batch의 모든 샘플에 더해진다. 그래서 편향 gradient는 batch 방향으로 합산한다.

```python
self.db = np.sum(dout, axis=0)
```

## 5. 자주 헷갈리는 점

`dx`, `dW`, `db`는 모두 loss에 대한 영향이다. 이름만 보면 단순한 변화량처럼 보이지만, 실제 의미는 다음과 같다.

```text
dx = dLoss/dx
dW = dLoss/dW
db = dLoss/db
```

또한 `dW`와 `db`는 이 층의 파라미터를 업데이트하기 위해 저장하고, `dx`는 이전 층으로 넘긴다는 차이가 있다.

## 6. 테스트 관점

`tests/test_affine.py`에서는 보통 다음 동작을 확인한다.

- forward 결과가 `x @ W + b`와 같은가
- backward에서 `dW`, `db`, `dx`가 올바른 shape를 가지는가
- `dW = x.T @ dout`이 맞는가
- `db = np.sum(dout, axis=0)`이 맞는가
- `dx = dout @ W.T`가 맞는가

`Affine`은 모든 신경망 층의 기본이므로 shape를 잘못 맞추면 이후 `ReLU`, `BatchNorm`, `Dropout`까지 모두 영향을 받는다.
