"""
MDP : unknown = model-free
prediction 방법 : MC, TD
그 중 MC prediction
- 목표 : 그리드 월드에서 4방향 random policy의 state별 value 구하기
- 구현 : environment, agent, main함수
- main함수 : 경험쌓는 부분(agent가 environment와 상호작용하며 데이터를 축적), 학습하는 부분(쌓인 경험을 통해 테이블을 업데이트)
"""
# 랜덤 에이전트 구현하기 위해
import random 
import numpy as np

# envrionment
class GridWorld():
    def __init__(self):
        self.x=0
        self.y=0
    
    # step 함수 가장 중요 : agent로부터 action을 받아서 state transition을 일으키고 reward를 정해주는 함수
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

        reward = -1 # 보상은 항상 -1로 고정
        done = self.is_done()
        return (self.x, self.y), reward, done

    def move_right(self):
        self.y += 1  
        if self.y > 3:
            self.y = 3
      
    def move_left(self):
        self.y -= 1
        if self.y < 0:
            self.y = 0
      
    def move_up(self):
        self.x -= 1
        if self.x < 0:
            self.x = 0
  
    def move_down(self):
        self.x += 1
        if self.x > 3:
            self.x = 3

    # 에피소드가 끝났는지 판별해주는 함수 
    def is_done(self):
        if self.x == 3 and self.y == 3:
            return True
        else :
            return False

    def get_state(self):
        return (self.x, self.y)
      
    # agent가 종료 상태에 도달했으면 다시 처음 상태로 돌려놓기 위해
    def reset(self):
        self.x = 0
        self.y = 0
        return (self.x, self.y)

# agent : 4방향 uniform random action을 선택하는 것뿐
class Agent():
    def __init__(self):
        pass        

    def select_action(self):
        coin = random.random()
        if coin < 0.25:
            action = 0
        elif coin < 0.5:
            action = 1
        elif coin < 0.75:
            action = 2
        else:
            action = 3
        return action


def main():
    env = GridWorld()
    agent = Agent()

    # 테이블
    data = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    gamma = 1.0
    reward = -1
    alpha = 0.001

    for k in range(50000):
        done = False
        history = []

        # agent가 경험을 쌓는 부분
        while not done:
            action = agent.select_action()
            (x,y), reward, done = env.step(action)
            history.append((x,y,reward))
        env.reset()

        # 쌓은 경험을 바탕으로 테이블 업데이트
        cum_reward = 0 # 리턴
        for transition in history[::-1]:
            x, y, reward = transition
            # MC 조금씩 업데이트 하는 버전 식
            data[x][y] = data[x][y] + alpha*(cum_reward-data[x][y])
            # 리턴 계산해줌
            cum_reward = reward + gamma*cum_reward  # 책에 오타가 있어 수정하였습니다
            
    for row in data:
        print(row)

if __name__ == '__main__':
    main()
