"""
DQN : Deep Q Network
2015년 구글 딥마인드 딥Q러닝 이용해 고전 비디오게임 아타리게임을 인간보다 더 잘하는 agent를 학습한 결과발표
이때 쓴 알고리즘 DQN
DQN : 뉴럴넷을 이용하여 표현한 Q함수를 강화하는 것
"""
import gym
import collections # 리플레이 버퍼 - deque(선입선출)
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000 # 원래 DQN 논문에서는 리플레이 버퍼 크기 100만
batch_size    = 32

class ReplayBuffer():
    """
    경험을 저장하는 리플레이 버퍼
    """
    def __init__(self):
        # 버퍼가 꽉 찬 상태에서 추가로 데이터가 들어오면, 자동으로 가장 먼저 들어온 데이터가 버퍼에서 밀려남
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        """
        미니 배치 크기 n만큼 샘플링
        """
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s) # state : 보통 벡터임(카트의 위치, 속도, 막대의 각도, 각속도), shape=(batch_size, 4)
            a_lst.append([a]) # a: 스칼라 -> [a] -> torch.tensor([a])로 변환하면 shape=(batch_size, 1) -> gather(1, a) 연산 수행가능
            r_lst.append([r]) # r: 스칼라 -> [r] -> torch.tensor([r])로 변환하면 shape=(batch_size, 1)
            s_prime_lst.append(s_prime) # s_prime : next state, shape=(batch_size, 4)
            done_mask_lst.append([done_mask]) # done_mask: 스칼라 -> [done_mask] -> torch.tensor([done_mask])로 변환하면 shape=(batch_size, 1)

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        """
        현재 버퍼에 저장된 transition 개수 반환
        """
        return len(self.buffer)

class Qnet(nn.Module):
    """
    - input : state 벡터
    - return : 모든 액션에 대한 각 액션의 value인 q(s,a)값 리턴. CartPole의 액션은 2개이므로, output 차원은 2
    """
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # 마지막에 relu 들어가지않음 주의 : 맨 마지막 아웃풋은 q 벨류여서, 어느 값이든 취할 수 있어서, 양수만 리턴하는 relu 넣어주면 안됨
        return x
    
    # 실제로 실행할 액션을 eps-greedy로 선택해주는 역할
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1) # 랜덤액션
        else : 
            return out.argmax().item() # Q값이 가장 높은 액션
            
def train(q, q_target, memory, optimizer):
    """
    - 에피소드가 끝날때마다 train 함수 호출
    - 한번 호출될 때마다 10개의 미니배치를 뽑아 10번 업데이트

    - q_target 네트워크 : 정답지를 계산할 때 쓰이는 네트워크 (학습대상 아님)
    - q 네트워크의 파라미터만 업데이트 대상임을 optimizer에게 알려줘야함
    """
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out = q(s) # (batch size, action dim) : 모든 액션의 Q값
        # a : (batch size, 1) 선택한 액션 인덱스
        q_a = q_out.gather(1,a) # (batch size, 1) 선택한 action의 q값 (열방향에서 a가 가리키는 인덱스를 선택)
        """
        q_target(s_prime) : (batch size, action dim)
        .max(1) : 열방향(액션차원)에서 가장 큰 값을 찾음
        [0] : 최대값 (각 state에서 선택가능한 모든 액션 중 최대값)
        .unsqueeze(1): (batch size,) -> (batch size, 1)
        """
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1) # max_q_prime 계산시, Q 네트워크가 아니라, target network가 호출되는 점 # 다음 state에서 최대 Q값 계산
        target = r + gamma * max_q_prime * done_mask # 벨만방정식적용
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward() # 실제 gradient 계산
        optimizer.step() # update 

def main():
    """
    CartPole : 카트를 잘 밀어서 막대가 넘어지지 않도록 균형을 잡는 문제
    - action : (왼쪽으로 밀기, 오른쪽으로 밀기)
    - reward : step마다 +1
    - state : (카트의 위치, 카트의 속도, 막대의 각도, 막대의 각속도)
    """
    env = gym.make('CartPole-v1') # 환경
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict()) # Q네트워크의 파라미터들을 타깃 네트워크로 복사할때 nn.Module에서 상속받아서 load_state_dict함수로 구현가능 # 타겟 네트워크 초기화
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        s, _ = env.reset() # 새로운 에피소드 시작하는 역할, 초기 state 반환
        done = False

        # 1개의 에피소드 종료될 때까지 경험을 쌓기
        while not done:
            # 하나의 에피소드가 끝날 때까지 액션 선택하고 실행하여 얻은 데이터를 리플레이 버퍼로 반복해서 보냄
            """
            s : 환경에서 얻은 넘파이 배열 (4,)
            torch.from_numpy(s) : 베열변환(dtype=torch.float64)
            float() : PyTorch에서 dtype=float32가 기본이므로 변경(float64 -> float32)
            torch.from_numpy(s).float() : (4,) PyTorch 텐서

            a : 스칼라
            """
            a = q.sample_action(torch.from_numpy(s).float(), epsilon) # 액션 선택
            s_prime, r, done, truncated, info = env.step(a) # 환경의 step함수 : 액션을 입력받아, 한 스텝 진행하고, 새로운 state, reward, 게임종료여부, 강제종료여부, 추가환경정보 리턴
            done_mask = 0.0 if done else 1.0 # 종료시 0, 아니면 1
            memory.put((s,a,r/100.0,s_prime, done_mask)) # 보상의 스케일이 너무 커서 조절하려고 100으로 나눔 # 리플레이 버퍼에 저장
            s = s_prime # 다음 state로 갱신

            score += r
            if done:
                break
    
        # 에피소드가 끝날 때마다, train 함수 호출해서 파라미터 업데이트 : 리플레이 버퍼에 데이터가 충분히 쌓이지 않았을 경우 학습을 진행하면 학습이 치우칠 수 있으므로, 2000개 이상 쌓였을 때부터 학습 진행
        if memory.size()>2000:
            train(q, q_target, memory, optimizer)

        # 에피소드가 10개 끝날 때마다 가장 최근 10개 에피소드의 보상 총합의 평균 프린트
        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict()) # q 네트워크 파라미터 q_target 네트워크로 복사(타겟 네트워크 업데이트)
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
    env.close() # 환경 종료 : 메모리 누수 방지

if __name__ == '__main__':
    main()