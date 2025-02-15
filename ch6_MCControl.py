"""
MDP : unknown = model-free
Control 방법 : MC Control, SARSA, Q러닝
그 중 MC Control
- Policy evaluation : 한 에피소드 경험 쌓고 경험한 데이터로 테이블 값 업데이트
- Policy improvement : 업데이트된 테이블을 이용해 eps-greedy policy 만듦
"""
import random
import numpy as np

class GridWorld():
    def __init__(self):
        self.x=0
        self.y=0
    
    def step(self, a):
        # 0번 액션: 왼쪽, 1번 액션: 위, 2번 액션: 오른쪽, 3번 액션: 아래쪽
        if a==0:
            self.move_left()
        elif a==1:
            self.move_up()
        elif a==2:
            self.move_right()
        elif a==3:
            self.move_down()

        reward = -1  # 보상은 항상 -1로 고정
        done = self.is_done()
        return (self.x, self.y), reward, done

    def move_left(self):
        if self.y==0:
            pass
        elif self.y==3 and self.x in [0,1,2]:
            pass
        elif self.y==5 and self.x in [2,3,4]:
            pass
        else:
            self.y -= 1

    def move_right(self):
        if self.y==1 and self.x in [0,1,2]:
            pass
        elif self.y==3 and self.x in [2,3,4]:
            pass
        elif self.y==6:
            pass
        else:
            self.y += 1
      
    def move_up(self):
        if self.x==0:
            pass
        elif self.x==3 and self.y==2:
            pass
        else:
            self.x -= 1

    def move_down(self):
        if self.x==4:
            pass
        elif self.x==1 and self.y==4:
            pass
        else:
            self.x+=1

    def is_done(self):
        if self.x==4 and self.y==6: # 목표 지점인 (4,6)에 도달하면 끝난다
            return True
        else:
            return False
      
    def reset(self):
        self.x = 0
        self.y = 0
        return (self.x, self.y)

class QAgent():
    def __init__(self):
        # q벨류를 저장하는 변수. 모두 0으로 초기화
        # q_table[x,y,a] : state(x,y)에서 액션 a의 Q값
        # update_table에서 Q값이 계속 업데이트됨
        self.q_table = np.zeros((5, 7, 4)) # 테이블 크기 (5, 7), action의 개수 4
        self.eps = 0.9 
        self.alpha = 0.01
        
    def select_action(self, s):
        # eps-greedy로 액션을 선택
        x, y = s
        coin = random.random() # 0.0 이상 1.0 미만의 실수를 무작위로 생성
        if coin < self.eps:
            # eps 확률로 exploration
            action = random.randint(0,3)
        else:
            # 1- eps확률로 가장 높은 Q값을 가진 액션 선택(현재 Q 테이블에서 가장 좋은 액션 선택)
            action_val = self.q_table[x,y,:] # 현재 상태 (x, y)에서 모든 액션들의 Q값을 가져와 비교
            action = np.argmax(action_val)
        return action

    def update_table(self, history):
        # 한 에피소드에 해당하는 history를 입력으로 받아 q 테이블의 값을 업데이트 한다
        cum_reward = 0
        for transition in history[::-1]:
            s, a, r, s_prime = transition
            x,y = s
            # 몬테 카를로 방식을 이용하여 업데이트.
            self.q_table[x,y,a] = self.q_table[x,y,a] + self.alpha * (cum_reward - self.q_table[x,y,a])
            cum_reward = cum_reward + r 

    def anneal_eps(self):
        self.eps -= 0.03
        self.eps = max(self.eps, 0.1) # 0.9부터 0.1까지 선형적으로 줄어듦

    def show_table(self):
        """
        학습이 끝난 후, state별로 q(s,a)의 값이 가장 큰 액션 뽑아서 보여주는 함수
        """
        q_lst = self.q_table.tolist()
        # data : Q값을 직접 저장하는게 아니라, 각 state에서 가장 Q값이 높은 액션을 저장
        data = np.zeros((5,7))
        for row_idx in range(len(q_lst)): # 행반복
            row = q_lst[row_idx] # 현재 행 가져오기
            for col_idx in range(len(row)): # 열반복
                col = row[col_idx] # 현재 위치 (x,y)의 Q값 리스트 (4개의 액션에 대한 Q값)
                action = np.argmax(col) # 가장 Q값이 높은 액션 선택
                data[row_idx, col_idx] = action # 해당 액션을 data 테이블에 액션 인덱스 저장
        print(data)
      
def main():
    env = GridWorld()
    agent = QAgent()

    for n_epi in range(1000): # 총 1,000 에피소드 동안 학습
        done = False
        history = []

        s = env.reset()
        while not done: # 한 에피소드가 끝날 때 까지
            a = agent.select_action(s)
            s_prime, r, done = env.step(a)
            history.append((s, a, r, s_prime))
            s = s_prime
        agent.update_table(history) # 히스토리를 이용하여 에이전트를 업데이트
        agent.anneal_eps()

    agent.show_table() # 학습이 끝난 결과를 출력

if __name__ == '__main__':
    main()