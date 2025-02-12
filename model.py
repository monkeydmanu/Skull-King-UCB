import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import logging
import torch.nn.functional as F

"""
# Configuration du deuxième logger pour écrire dans test1.log
logger_model = logging.getLogger('logger_model1')
logger_model.setLevel(logging.INFO)
# Handler pour écrire dans le fichier test1.log
handler_model = logging.FileHandler('model_test.log', mode='w')  # Ouvre en mode 'w' pour écraser à chaque exécution
handler_model.setFormatter(logging.Formatter('%(message)s'))
logger_model.addHandler(handler_model)
"""

def choix_indice_aleatoire_parmi_indice_carte_dispo(indice_carte_dispo):
    res = random.choice(indice_carte_dispo)
    return res

def choix_indice_aleatoire_parmi_nb_pli_max(nb_pli_max_possible):
    res = random.randrange(nb_pli_max_possible+1)
    return res

def choix_indice_aleatoire(n_actions):
    res = random.randrange(n_actions)
    return res # de 0 à 21 non inclus donc 21 actions possibles

# Création du modèl
class DeepQNetwork(nn.Module):
    def __init__(self, input_shape, hidden_units, n_actions, lr, device):
        super().__init__()
        self.input_shape = input_shape[0] if isinstance(input_shape, tuple) else input_shape # changer d'approche si on passe des tuples pour de vrai
        self.hidden_units = hidden_units[0] if isinstance(hidden_units, tuple) else hidden_units
        self.n_actions = n_actions[0] if isinstance(n_actions, tuple) else n_actions
        self.device = device

        self.block1 = nn.Sequential(
            nn.Linear(in_features=self.input_shape, out_features=self.hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_units, out_features=self.hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_units, out_features=self.n_actions)
        ).to(self.device)
        """
            nn.Linear(in_features=self.hidden_units, out_features=self.hidden_units),
            nn.ReLU(),
        """
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        # IL FAUT INSTANCIER SIGMOID si on veut l'utiliser en dessous je pense

    def forward(self, x):
        return self.block1(x) # nn.Softmax(dim=0)(self.block1(x))
    #nn.Softmax(dim=-1)
# utiliser SOFTMAX a la place de sigmoid



class BaseAgent:
    def __init__(self, gamma, lr, input_dims, n_actions_jouer, n_actions_predire, max_mem_size=10000):
        self.illegal_moves_count_predire = 0
        self.illegal_moves_count = 0
        self.illegal_moves_per_game = []
        self.illegal_moves_per_game_predire = []
        self.loss_learn = []
        self.loss_learn_seul = []
        self.current_loss_learn_sum = 0
        self.current_loss_learn_seul_sum = 0
        self.gamma = gamma
        self.lr = lr
        self.mem_size = max_mem_size
        self.mem_cntr = 0
        self.mem_cntr_predire = 0
        device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.n_actions_jouer = n_actions_jouer
        self.n_actions_predire = n_actions_predire
        self.input_dims = input_dims
        self.c_history = []
        self.nb_action_tot_jouer = 0
        self.nb_action_tot_predire = 0

        # Modèle pour prédire
        self.Q_eval_jouer = DeepQNetwork(input_shape=input_dims, hidden_units=128, n_actions=self.n_actions_jouer, lr=self.lr, device=device)
        self.target_dqn_jouer = DeepQNetwork(input_shape=input_dims, hidden_units=128, n_actions=self.n_actions_jouer, lr=self.lr, device=device)

        # Modèle pour jouer
        self.Q_eval_predire = DeepQNetwork(input_shape=input_dims, hidden_units=128, n_actions=self.n_actions_predire, lr=self.lr, device=device)
        self.target_dqn_predire = DeepQNetwork(input_shape=input_dims, hidden_units=128, n_actions=self.n_actions_predire, lr=self.lr, device=device)

        self.action_counts_jouer = np.zeros(n_actions_jouer)
        self.action_counts_predire = np.zeros(n_actions_predire)

        # Mémoires pour les transitions
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)

        self.state_memory_predire = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory_predire = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory_predire = np.zeros(self.mem_size, dtype=np.int32)


    def reset(self):
        self.illegal_moves_per_game.append(self.illegal_moves_count)
        self.illegal_moves_per_game_predire.append(self.illegal_moves_count_predire)
        self.illegal_moves_count = 0
        self.illegal_moves_count_predire = 0
        # Réinitialiser les compteurs de mémoire et d'itérations
        self.mem_cntr = 0
        self.mem_cntr_predire = 0
        # Réinitialiser les mémoires
        self.state_memory.fill(0)
        self.new_state_memory.fill(0)
        self.action_memory.fill(0)

    def update_c(self):
        self.c_history.append(self.c)

    def save(self, filename, q_eval):
        checkpoint = {
            'model_state_dict': q_eval.state_dict(),
            'optimizer_state_dict': q_eval.optimizer.state_dict(),
            'mem_cntr': self.mem_cntr,
            'mem_cntr_predire': self.mem_cntr_predire,
            'count_action_jouer': self.action_counts_jouer,
            'count_action_predire': self.action_counts_predire,
            'nb_action_tot_jouer': self.nb_action_tot_jouer,
            'nb_action_tot_predire': self.nb_action_tot_predire,
            'c': self.c
        }
        T.save(checkpoint, filename)

        print(f"Agent saved successfully to {filename}.")


    def load(self, filename, q_eval):
        checkpoint = T.load(filename)
        q_eval.load_state_dict(checkpoint['model_state_dict'])
        q_eval.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.mem_cntr = checkpoint['mem_cntr']
        self.mem_cntr_predire = checkpoint['mem_cntr_predire']
        self.action_counts_jouer = checkpoint['count_action_jouer']
        self.action_counts_predire = checkpoint['count_action_predire']
        self.nb_action_tot_jouer = checkpoint['nb_action_tot_jouer']
        self.nb_action_tot_predire = checkpoint['nb_action_tot_predire']
        self.c = checkpoint['c']

        print(f"Agent loaded successfully from {filename}.")

    def clean_illegal_transitions(self):

        self.illegal_moves_count += 1
        if self.mem_cntr > 0:
            # Calculer l'index du dernier élément
            index = (self.mem_cntr - 1) % self.mem_size

            # Déplacer les éléments suivants vers la gauche pour combler le vide
            self.state_memory[index] = np.zeros_like(self.state_memory[index])
            self.new_state_memory[index] = np.zeros_like(self.new_state_memory[index])
            self.action_memory[index] = 0

            # Réduire le compteur de mémoire
            self.mem_cntr -= 1
            

    def clean_illegal_transitions_predire(self):

        self.illegal_moves_count_predire += 1
        if self.mem_cntr_predire > 0:
            # Calculer l'index du dernier élément
            index = (self.mem_cntr_predire - 1) % self.mem_size

            # Déplacer les éléments suivants vers la gauche pour combler le vide
            self.state_memory_predire[index] = np.zeros_like(self.state_memory_predire[index])
            self.new_state_memory_predire[index] = np.zeros_like(self.new_state_memory_predire[index])
            self.action_memory_predire[index] = 0

            # Réduire le compteur de mémoire
            self.mem_cntr_predire -= 1

    def store_transition(self, state, action, state_):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = np.array(state, dtype=np.float32)
        self.new_state_memory[index] = np.array(state_, dtype=np.float32)
        self.action_memory[index] = action
        self.mem_cntr += 1

    def store_transition_predire(self, state, action, state_):
        index = self.mem_cntr_predire % self.mem_size
        self.state_memory_predire[index] = np.array(state, dtype=np.float32)
        self.new_state_memory_predire[index] = np.array(state_, dtype=np.float32)
        self.action_memory_predire[index] = action
        self.mem_cntr_predire += 1

# Upper Confidence Bound (UCB)

class Agent(BaseAgent):
    def __init__(self, gamma, lr, input_dims, n_actions_jouer, n_actions_predire, max_mem_size=10000):
        super().__init__(gamma, lr, input_dims, n_actions_jouer, n_actions_predire, max_mem_size)
        self.c = 1

    def reset_game_loss(self):
        self.current_loss_learn_sum = 0
        self.current_loss_learn_seul_sum = 0

    def update_value_loss(self):
        self.loss_learn.append(self.current_loss_learn_sum)
        self.loss_learn_seul.append(self.current_loss_learn_seul_sum)

    def choose_action(self, observation, indice_carte_dispo=None, nb_prediction_max=None):
        # UCB
        state = T.tensor(np.array(observation), dtype=T.float32).to(self.Q_eval_jouer.device)

        if self.nb_action_tot_jouer % 200 == 0:
            print(f"\n{self.nb_action_tot_jouer = }\n{self.nb_action_tot_predire = }\n{self.action_counts_jouer = }\n{self.action_counts_predire = }")
        if indice_carte_dispo:
            actions = self.Q_eval_jouer.forward(state)
            q_values = F.softmax(actions, dim=0).detach().cpu().numpy()
            ucbs = [q_values[i] + self.c * np.sqrt(np.log(self.nb_action_tot_jouer + 1) / ((self.action_counts_jouer[i] + 1))) for i in range(self.n_actions_jouer)]
            #action = np.argmax(ucbs)
            action = max(indice_carte_dispo, key=lambda i: ucbs[i].item())

            if self.nb_action_tot_jouer % 200 == 0:
                print(f"\n{q_values = }")
                print(f"\nPour jouer {indice_carte_dispo = }: {ucbs = }")
            self.action_counts_jouer[action] += 1
            self.nb_action_tot_jouer += 1
        else :
            actions = self.Q_eval_predire.forward(state)
            q_values = F.softmax(actions, dim=0).detach().cpu().numpy()
            ucbs = [q_values[i] + self.c * np.sqrt(np.log(self.nb_action_tot_predire + 1) / ((self.action_counts_predire[i] + 1))) for i in range(self.n_actions_predire)]
            action = max(range(nb_prediction_max), key=lambda i: ucbs[i].item())
            #action = np.argmax(ucbs)
            if self.nb_action_tot_predire % 20 == 0:
                print(f"\n{q_values = }")
                print(f"Pour predire {nb_prediction_max = } : {ucbs = }")
            self.action_counts_predire[action] += 1
            self.nb_action_tot_predire += 1

        
        if self.c < 0.2:
            self.c = 0
        else:
            self.c = self.c * np.exp(-(1e-6)*self.mem_cntr)
        # print(f"---------------- {self.c = } -------------------")

        self.update_c()

        return action
    
    # dans learn j'appprends avec la meilleurs des 10 valeurs qui m'intéressent
    def learn_tout(self, reward):

        # POUR APPRENDRE POUR LES PLIS JOUES
        q_eval_list = []
        q_target_list = []

        # Indices de l'état sélectionné
        suite_indice_state_selectionner = list(range(55))

        # Optimisation par lots
        self.Q_eval_jouer.optimizer.zero_grad()

        for index in suite_indice_state_selectionner:
            # État et nouvel état pour chaque transition
            state = T.tensor(self.state_memory[index], dtype=T.float32).to(self.Q_eval_jouer.device)
            new_state = T.tensor(self.new_state_memory[index], dtype=T.float32).to(self.Q_eval_jouer.device)
            action = self.action_memory[index]

            # Obtenir la prédiction de Q pour l'état actuel et le nouvel état
            current_q = self.Q_eval_jouer.forward(state)
            next_q = self.target_dqn_jouer.forward(new_state)

            with T.no_grad():
                
                target = reward + self.gamma * T.max(next_q)

            target_q = self.target_dqn_jouer.forward(state)
            target_q[action] = target

            # Ajouter à la liste
            q_eval_list.append(current_q)
            q_target_list.append(target_q)


        # Création de tenseurs à partir des listes
        q_eval_tensor = T.stack(q_eval_list)
        q_target_tensor = T.stack(q_target_list)

        # Calcul de la perte
        loss = self.Q_eval_jouer.loss(q_eval_tensor, q_target_tensor).to(self.Q_eval_jouer.device)

        # Backpropagation et optimisation
        loss.backward()
        self.Q_eval_jouer.optimizer.step()


        # POUR APPRENDRE SUR LES PREDICTIONS
        q_eval_list = []
        q_target_list = []

        # Indices de l'état sélectionné
        suite_indice_state_selectionner = list(range(10))

        # Optimisation par lots
        self.Q_eval_predire.optimizer.zero_grad()

        for index in suite_indice_state_selectionner:
            # État et nouvel état pour chaque transition
            state = T.tensor(self.state_memory_predire[index], dtype=T.float32).to(self.Q_eval_predire.device)
            new_state = T.tensor(self.new_state_memory_predire[index], dtype=T.float32).to(self.Q_eval_predire.device)
            action = self.action_memory_predire[index]

            # Obtenir la prédiction de Q pour l'état actuel et le nouvel état
            current_q = self.Q_eval_predire.forward(state)
            next_q = self.target_dqn_predire.forward(new_state)

            with T.no_grad():
                
                target = reward + self.gamma * T.max(next_q)

            target_q = self.target_dqn_predire.forward(state)
            target_q[action] = target

            # Ajouter à la liste
            q_eval_list.append(current_q)
            q_target_list.append(target_q)

        

        # Création de tenseurs à partir des listes
        q_eval_tensor = T.stack(q_eval_list)
        q_target_tensor = T.stack(q_target_list)

        # Calcul de la perte
        loss = self.Q_eval_predire.loss(q_eval_tensor, q_target_tensor).to(self.Q_eval_predire.device)

        # Backpropagation et optimisation
        loss.backward()
        self.Q_eval_predire.optimizer.step()


    # dans learn j'appprends avec la meilleurs des 10 valeurs qui m'intéressent
    def learn(self, reward):

        q_eval_list = []
        q_target_list = []

        indice = (self.mem_cntr - 1) % self.mem_size if self.mem_cntr > 0 else 0
        nb_manche = self.state_memory[indice][self.input_dims[0] - 3]
        suite_indice_state_selectionner = list(range(self.mem_cntr - int(nb_manche), self.mem_cntr)) if self.mem_cntr > 0 else []

        # print(f"Pour learn : {reward = }")

        # Charger les batchs d'états et de nouvelles transitions
        state_batch = T.tensor(self.state_memory[suite_indice_state_selectionner], dtype=T.float32).to(self.Q_eval_jouer.device)
        new_state_batch = T.tensor(self.new_state_memory[suite_indice_state_selectionner], dtype=T.float32).to(self.Q_eval_jouer.device)
        action_batch = self.action_memory[suite_indice_state_selectionner]
        batch_index = np.arange(nb_manche, dtype=np.int32)

        q_eval = self.Q_eval_jouer.forward(state_batch)
        q_next = self.Q_eval_jouer.forward(new_state_batch)

        self.Q_eval_jouer.optimizer.zero_grad()

        for index in suite_indice_state_selectionner:
            # État et nouvel état pour chaque transition
            state = T.tensor(self.state_memory[index], dtype=T.float32).to(self.Q_eval_jouer.device)
            new_state = T.tensor(self.new_state_memory[index], dtype=T.float32).to(self.Q_eval_jouer.device)
            action = self.action_memory[index]

            # Obtenir la prédiction de Q pour l'état actuel et le nouvel état
            current_q = self.Q_eval_jouer.forward(state)
            next_q = self.target_dqn_jouer.forward(new_state)

            with T.no_grad():
                
                target = reward + self.gamma * T.max(next_q)

            target_q = self.target_dqn_jouer.forward(state)
            target_q[action] = target

            # print(f"{current_q = }\n {target_q = }")

            # Ajouter à la liste
            q_eval_list.append(current_q)
            q_target_list.append(target_q)

        

        # Création de tenseurs à partir des listes
        q_eval_tensor = T.stack(q_eval_list)
        q_target_tensor = T.stack(q_target_list)

        # print(f"{q_eval_tensor[-1] = }")

        # Calcul de la perte
        loss_action = self.Q_eval_jouer.loss(q_eval_tensor, q_target_tensor).to(self.Q_eval_jouer.device)

        self.current_loss_learn_sum += loss_action.item()
        loss_action.backward()
        self.Q_eval_jouer.optimizer.step()

    # dans learn_seul j'appprends avec la meilleurs des 20 valeurs
    def learn_seul(self, reward, plis_gagnes=None, is_predire=False):

        # print("LEARN SEUL")
        # print(f"{reward = }")
        if is_predire and plis_gagnes is None: # à enlever car juste pour dire que c'est pas un coup illégal
            indice = (self.mem_cntr_predire - 1) % self.mem_size if self.mem_cntr_predire > 0 else 0
        elif plis_gagnes is None: # à enlever, pour dire que la prédiction n'est pas illégal
            indice = (self.mem_cntr - 1) % self.mem_size if self.mem_cntr > 0 else 0
        else: # pour la fin de chaque manche, pour comparer le nombre de plis prédis avec le nombre de plis gagnés
            indice = (self.mem_cntr_predire - 1) % self.mem_size if self.mem_cntr_predire > 0 else 0

        """
        # pour le learn_seul à la fin de la manche
        # on prend l'état sur lequel on a prédit pour le comparé avec le nb de plis gagnes
        nb_manche = int(self.state_memory[indice][self.input_dims[0] -3])
        # print(f"{nb_manche = }")
        if self.mem_cntr > 0:
            indice_state_prediction_debut = self.mem_cntr - int(nb_manche)
        else:
            indice_state_prediction_debut = 0
        """

        if is_predire and plis_gagnes is None:
            self.Q_eval_predire.optimizer.zero_grad()
        elif plis_gagnes is None:
            self.Q_eval_jouer.optimizer.zero_grad()
        else:
            self.Q_eval_predire.optimizer.zero_grad()

        # pour le learn_seul d'une mauvaise prédiction
        if is_predire and plis_gagnes is None:
            # print("mauvaise prédiction")
            state_batch = T.tensor(self.state_memory_predire[indice]).to(self.Q_eval_predire.device)
            new_state_batch = T.tensor(self.new_state_memory_predire[indice]).to(self.Q_eval_predire.device)
            action_batch = self.action_memory_predire[indice]

            # Obtenir la prédiction de Q pour l'état actuel et le nouvel état
            current_q = self.Q_eval_predire.forward(state_batch)
            next_q = self.target_dqn_predire.forward(new_state_batch)

            with T.no_grad():
                
                target = reward + self.gamma * T.max(next_q)

            target_q = self.target_dqn_predire.forward(state_batch)
            target_q[action_batch] = target

        # pour learn_seul d'une mauvais action
        elif plis_gagnes is None:
            # print("mauvais action")
            state_batch = T.tensor(self.state_memory[indice]).to(self.Q_eval_jouer.device)
            new_state_batch = T.tensor(self.new_state_memory[indice]).to(self.Q_eval_jouer.device)
            action_batch = self.action_memory[indice]
            
            # Obtenir la prédiction de Q pour l'état actuel et le nouvel état
            current_q = self.Q_eval_jouer.forward(state_batch)
            next_q = self.target_dqn_jouer.forward(new_state_batch)

            with T.no_grad():
                
                target = reward + self.gamma * T.max(next_q)

            target_q = self.target_dqn_jouer.forward(state_batch)
            target_q[action_batch] = target

        # pour le learn_seul de la prédiction à la fin
        else:
            # print("prédiction à la fin")
            state_batch = T.tensor(self.state_memory_predire[indice]).to(self.Q_eval_predire.device)
            action_batch = self.action_memory_predire[indice]

            current_q = self.Q_eval_predire.forward(state_batch)

            with T.no_grad():
                target = reward + self.gamma * T.tensor(1, dtype=T.float32).to(self.Q_eval_predire.device)

            target_q = T.zeros_like(current_q)
            target_q[action_batch] = target

        # print(f"{current_q = }")
        # Calcul de la perte
        

        if is_predire and plis_gagnes is None:
            loss = self.Q_eval_predire.loss(current_q, target_q).to(self.Q_eval_predire.device)
            self.current_loss_learn_seul_sum += loss.item()
            loss.backward()
        elif plis_gagnes is None:
            loss = self.Q_eval_jouer.loss(current_q, target_q).to(self.Q_eval_jouer.device)
            self.current_loss_learn_seul_sum += loss.item()
            loss.backward()
        else:
            loss = self.Q_eval_predire.loss(current_q, target_q).to(self.Q_eval_predire.device)
            self.current_loss_learn_seul_sum += loss.item()
            loss.backward()

        if is_predire and plis_gagnes is None:
            self.Q_eval_predire.optimizer.step()
        elif plis_gagnes is None:
            self.Q_eval_jouer.optimizer.step()
        else:
            self.Q_eval_predire.optimizer.step()


    # Visualisation des graphiques
    def plot_all_graphs(self, joueur):
        # Initialisation de la figure avec 2 lignes et 2 colonnes
        plt.figure(figsize=(14, 10))

        # 1er graphique : Quantité de coups illégaux par jeu
        plt.subplot(3, 2, 1)
        plt.plot(self.illegal_moves_per_game, marker='o', color='r')
        plt.title(f'Quantité de coups illégaux par jeu pour l\'{joueur.nom}')
        plt.xlabel('Jeu')
        plt.ylabel('Coups illégaux')

        # 2eme graphique : Quantité de coups illégaux prédiction par jeu
        plt.subplot(3, 2, 2)
        plt.plot(self.illegal_moves_per_game_predire, marker='o', color='r')
        plt.title(f'Quantité de coups illégaux prédiction par jeu pour l\'{joueur.nom}')
        plt.xlabel('Jeu')
        plt.ylabel('Coups illégaux prédiction')

        # 3ème graphique : Perte pour learn
        plt.subplot(3, 2, 3)
        plt.plot(self.loss_learn, label='Loss Learn', color='b')
        plt.title(f'Perte en fonction du temps pour l\'{joueur.nom}')
        plt.xlabel('Itérations')
        plt.ylabel('Perte')
        plt.legend()

        # 4ème graphique : Perte pour learn_seul
        plt.subplot(3, 2, 4)
        plt.plot(self.loss_learn_seul, label='Loss Learn Seul', color='g')
        plt.title(f'Perte learn_seul en fonction du temps pour l\'{joueur.nom}')
        plt.xlabel('Itérations')
        plt.ylabel('Perte')
        plt.legend()

        # 5ème graphique : Évolution des c
        plt.subplot(3, 2, 5)
        plt.plot(self.c_history, label='Valeur de c', color='b')
        plt.title('Évolution de c en fonction du temps')
        plt.xlabel('Itérations')
        plt.ylabel('Valeur de c')
        plt.legend()
        plt.grid(True)


        # Affichage de tous les graphiques
        plt.tight_layout()
        plt.show()
