# -*- coding: utf-8 -*-
"""학습 루프, 평가, 시각화 함수 모음."""

import numpy as np

from losses import cross_entropy_loss


def train(
    model,
    optimizer,
    x_train,
    y_train,
    epochs=20,
    batch_size=128,
    x_val=None,
    y_val=None,
    verbose=False,
):
    """
    미니배치 학습 루프.

    한 배치마다 Forward -> Loss -> Backward -> Optimizer 업데이트 순서로 진행합니다.
    교육생은 이 함수에서 "예측값을 만들고, 손실을 계산하고, gradient로 파라미터를 바꾸는"
    전체 흐름을 확인할 수 있습니다.

    Args:
        x_val, y_val: epoch마다 함께 확인할 검증/테스트 데이터
        verbose: True이면 epoch별 loss와 accuracy를 출력

    Returns:
        loss_history: epoch별 평균 손실 리스트
    """
    # TODO: epoch마다 데이터를 섞고, batch 단위로 forward/loss/backward/update를 수행하세요.
    # 힌트: Softmax + CrossEntropy 결합 gradient는 y_pred copy에서 정답 위치에 1을 빼서 만듭니다.

    loss_history = []

    for epoch in range(epochs):
        # 매 epoch마다 데이터 순서를 섞어, 특정 배치 구성에만 맞춰 학습되는 것을 막습니다.
        indices = np.random.permutation(len(x_train))
        x_train = x_train[indices]
        y_train = y_train[indices]
        epoch_loss = 0

        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            # 마지막 배치는 batch_size보다 작을 수 있으므로 실제 배치 크기를 사용합니다.
            current_batch_size = x_batch.shape[0]

            y_pred = model.forward(x_batch, train=True)
            loss = cross_entropy_loss(y_pred, y_batch)

            # Softmax + CrossEntropy의 gradient: 예측 확률에서 정답 클래스 위치만 1을 뺍니다.
            dout = y_pred.copy()
            dout[np.arange(current_batch_size), y_batch] -= 1
            dout /= current_batch_size

            model.backward(dout)
            optimizer.update(model.params, model.grads)

            # loss는 배치 평균이므로, 샘플 수를 곱해 epoch 전체 평균을 계산할 준비를 합니다.
            epoch_loss += loss * current_batch_size

        avg_loss = epoch_loss / len(x_train)
        loss_history.append(avg_loss)

        if verbose:
            # 추론 모드로 전체 학습 데이터 정확도를 계산해 epoch별 학습 상태를 확인합니다.
            train_pred = model.predict(x_train)
            train_acc = np.mean(np.argmax(train_pred, axis=1) == y_train) * 100
            message = f"Epoch {epoch + 1:02d}/{epochs} - loss: {avg_loss:.4f} - train_acc: {train_acc:.2f}%"

            if x_val is not None and y_val is not None:
                val_pred = model.predict(x_val)
                val_acc = np.mean(np.argmax(val_pred, axis=1) == y_val) * 100
                message += f" - val_acc: {val_acc:.2f}%"

            print(message)
    return loss_history

def evaluate(model, x, y):
    """정확도(%)와 총 파라미터 수 반환."""
    y_pred = model.predict(x)
    accuracy = np.mean(np.argmax(y_pred, axis=1) == y) * 100
    total_params = sum(p.size for p in model.params.values())
    return accuracy, total_params


def plot_loss_history(loss_history):
    """손실 커브 그래프."""
    import matplotlib.pyplot as plt

    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()
