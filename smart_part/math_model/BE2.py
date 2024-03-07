import numpy as np


class BE2:
    def __init__(self, R1, W1, R2, W2, W):
        self.R1 = R1
        self.W1 = W1
        self.R2 = R2
        self.W2 = W2
        self.W = W

    def normalized(self, data):
        """
        计算概率分布
        :param data: 需要归一化的数组
        :return:
        """
        sum = 0
        for i in range(len(data)):
            sum += data[i]
        if sum == 1:
            return np.array(data)
        return np.array(data) / sum

    def min_max_operator(self, A, R):
        '''
        主因素突出型：M(Λ, V)
        利用最值算子合成矩阵
        :param A:评判因素权向量 A = (a1,a2 ,L,an )
        :param R:模糊关系矩阵 R
        :return:
        '''
        B = np.zeros((1, np.shape(R)[1]))
        for column in range(0, np.shape(R)[1]):
            list = []
            for row in range(0, np.shape(R)[0]):
                list.append(min(A[row], R[row, column]))
            B[0, column] = max(list)
        return B

    def min_add_operator(self, A, R):
        '''
        主因素突出型：M(Λ, +)
        先取小，再求和
        :param A:评判因素权向量 A = (a1,a2 ,L,an )
        :param R:模糊关系矩阵 R
        :return:
        '''
        B = np.zeros((1, np.shape(R)[1]))
        for column in range(0, np.shape(R)[1]):
            list = []
            for row in range(0, np.shape(R)[0]):
                list.append(min(A[row], R[row, column]))
            B[0, column] = np.sum(list)
        return B

    def mul_max_operator(self, A, R):
        '''
        加权平均型：M(*, +)
        利用乘法最大值算子合成矩阵
        :param A:评判因素权向量 A = (a1,a2 ,L,an )
        :param R:模糊关系矩阵 R
        :return:
        '''
        B = np.zeros((1, np.shape(R)[1]))
        for column in range(0, np.shape(R)[1]):
            list = []
            for row in range(0, np.shape(R)[0]):
                list.append(A[row] * R[row, column])
            B[0, column] = max(list)
        return B

    def mul_add_operator(self, A, R):
        '''
        加权平均型：M(*, +)
        先乘再求和
        :param A:评判因素权向量 A = (a1,a2 ,L,an )
        :param R:模糊关系矩阵 R
        :return:
        '''
        return np.matmul(self.W1, self.R1)

    def get(self, W, R):
        """
        :return: 获得var最大的
        """
        s = [self.normalized(self.min_max_operator(W, R).reshape(np.shape(R)[1])),
             self.normalized(self.min_add_operator(W, R).reshape(np.shape(R)[1])),
             self.normalized(self.mul_max_operator(W, R).reshape(np.shape(R)[1])),
             self.normalized(self.mul_add_operator(W, R).reshape(np.shape(R)[1]))]
        vars = []
        for i in range(len(s)):
            vars.append(s[i].var())

        i = np.argmax(vars)
        return s[i]

    def run(self):
        R = np.vstack([
            self.get(self.W1, self.R1),
            self.get(self.W2, self.R2),
        ])

        return np.dot(self.get(self.W, R), [1, 0.7, 0.5, 0])


if __name__ == '__main__':
    # 情绪的隶属度矩阵(待调整)
    mood_membership = np.array([
        [0.4, 0.3, 0.2, 0.1],  # 积极
        [0.6, 0.2, 0.1, 0.1],  # 中级
        [0.1, 0.1, 0.2, 0.6]  # 消极
    ])

    # 情绪的权重(也就是概率，经过神经网络得到)
    mood_weights = np.array([
        0.6, 0.3, 0.1
    ])

    # 转头角度的隶属度矩阵(待调整)
    angle_membership = np.array([
        [0.6, 0.2, 0.1, 0.1],  # 20-40
        [0.4, 0.3, 0.2, 0.1],  # 0-20
        [0.1, 0.1, 0.2, 0.6],  # 40-60
        [0.1, 0.1, 0.2, 0.6]  # >60
    ])

    # 转头角度的权重(也就是概率，经过神经网络得到，如果确切数值的话，就把其他置为0)
    angle_weight = np.array([
        0, 0, 1, 0
    ])

    weight = [0.3, 0.7]

    be2 = BE2(mood_membership, mood_weights, angle_membership, angle_weight, weight)
    score = be2.run()
    print("得分为：",score)
