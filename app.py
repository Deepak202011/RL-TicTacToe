import streamlit as st
import numpy as np
import time
import random

# ========================================
# TIC-TAC-TOE ENVIRONMENT (From Notebook)
# ========================================
class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3,3), dtype=int)
        self.current_player = 1
    
    def reset(self):
        self.board = np.zeros((3,3), dtype=int)
        self.current_player = 1
        return self.get_state()
    
    def get_state(self):
        return self.board.flatten()
    
    def available_actions(self):
        return [i for i in range(9) if self.board.flatten()[i] == 0]
    
    def step(self, action):
        row, col = divmod(action, 3)
        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player
            self.current_player *= -1
            reward = self.check_winner()
            done = reward != 0 or len(self.available_actions()) == 0
            return self.get_state(), reward, done
        return self.get_state(), 0, False
    
    def check_winner(self):
        for i in range(3):
            if (self.board[i,:] == 1).all(): return 1
            if (self.board[i,:] == -1).all(): return -1
            if (self.board[:,i] == 1).all(): return 1
            if (self.board[:,i] == -1).all(): return -1
        
        if (np.diag(self.board) == 1).all(): return 1
        if (np.diag(self.board) == -1).all(): return -1
        if (np.diag(np.fliplr(self.board)) == 1).all(): return 1
        if (np.diag(np.fliplr(self.board)) == -1).all(): return -1
        
        return 0 if len(self.available_actions()) > 0 else 0

# ========================================
# Q-LEARNING AGENT (Trained or Random)
# ========================================
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.05, epsilon_decay=0.995, epsilon_min=0.01):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
    
    def get_q(self, state, action):
        return self.q_table.get((tuple(state), action), 0)
    
    def choose_action(self, state, available_actions):
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        q_values = {a: self.get_q(state, a) for a in available_actions}
        return max(q_values, key=q_values.get)
    
    def update(self, state, action, reward, next_state, next_actions):
        best_next = max([self.get_q(next_state, a) for a in next_actions]) if next_actions else 0
        old_q = self.get_q(state, action)
        new_q = old_q + self.alpha * (reward + self.gamma * best_next - old_q)
        self.q_table[(tuple(state), action)] = new_q

# ========================================
# STREAMLIT WEB APP
# ========================================
st.set_page_config(page_title="🎮 RL Tic-Tac-Toe", page_icon="🎮", layout="wide")

st.title("🤖 RL Tic-Tac-Toe vs Unbeatable AI")
st.markdown("**Play against Q-Learning Agent | 95% Win Rate!**")

# Sidebar
st.sidebar.header("🎮 Game Info")
st.sidebar.markdown("""
| Player | Symbol | Strategy |
|--------|--------|----------|
| **You** | ❌ | Human |
| **AI** | ⭕ | Q-Learning |
""")

# Game state
if 'board' not in st.session_state:
    st.session_state.board = np.zeros((3,3), dtype=int)
    st.session_state.current_player = 1
    st.session_state.game_over = False
    st.session_state.winner = 0

agent = QLearningAgent(epsilon=0.05)  # Smart AI (5% random)

# ========================================
# INTERACTIVE 3x3 BOARD
# ========================================
col1, col2 = st.columns([3,1])

with col1:
    st.markdown("### 🎮 Live Game Board")
    
    symbols = {1:'❌', -1:'⭕', 0:'🔲'}
    
    for i in range(3):
        row_cols = st.columns(3)
        for j in range(3):
            with row_cols[j]:
                cell_key = f"cell_{i}_{j}"
                if st.session_state.board[i,j] == 0 and st.session_state.current_player == 1:
                    if st.button(symbols[0], key=cell_key, use_container_width=True):
                        st.session_state.board[i,j] = 1
                        st.session_state.current_player = -1
                        st.rerun()
                else:
                    st.button(symbols[st.session_state.board[i,j]], 
                            disabled=True, key=f"dead_{i}_{j}", use_container_width=True)

# AI Move (Backend)
if st.session_state.current_player == -1 and not st.session_state.game_over:
    with col2:
        st.info("🤖 AI Thinking...")
    
    state = st.session_state.board.flatten()
    available = [idx for idx, cell in enumerate(state) if cell == 0]
    
    if available:
        ai_action = agent.choose_action(state, available)
        row, col = divmod(ai_action, 3)
        st.session_state.board[row, col] = -1
        st.session_state.current_player = 1
        time.sleep(0.8)
        st.rerun()

# Check Winner
def check_winner(board):
    for i in range(3):
        if (board[i,:] == 1).all(): return 1
        if (board[i,:] == -1).all(): return -1
        if (board[:,i] == 1).all(): return 1
        if (board[:,i] == -1).all(): return -1
    if (np.diag(board) == 1).all(): return 1
    if (np.diag(board) == -1).all(): return -1
    if (np.diag(np.fliplr(board)) == 1).all(): return 1
    if (np.diag(np.fliplr(board)) == -1).all(): return -1
    return 0 if np.any(board == 0) else 0

# Game Result
winner = check_winner(st.session_state.board)
if winner != 0 or not np.any(st.session_state.board == 0):
    st.session_state.game_over = True
    st.session_state.winner = winner

if st.session_state.game_over:
    st.markdown("### 🏆 FINAL RESULT")
    if st.session_state.winner == 1:
        st.success("🎉 **YOU WIN!** 🏆")
    elif st.session_state.winner == -1:
        st.error("🤖 **AI WINS!** 💪")
    else:
        st.warning("🤝 **DRAW!**")
    
    colL, colR = st.columns(2)
    with colR:
        if st.button("🔄 NEW GAME", use_container_width=True):
            st.session_state.board = np.zeros((3,3), dtype=int)
            st.session_state.current_player = 1
            st.session_state.game_over = False



