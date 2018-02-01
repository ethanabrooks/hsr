from os.path import join

import numpy as np
from gym import spaces
from itertools import combinations, product


from environment.base import BaseEnv


class Arm2TouchEnv(BaseEnv):
    def __init__(self, continuous, max_steps, geofence=.08, history_len=1, neg_reward=True,
                 action_multiplier=1):

        BaseEnv.__init__(self,
                         geofence=geofence,
                         max_steps=max_steps,
                         xml_filepath=join('models', 'arm2touch', 'world.xml'),
                         history_len=history_len,
                         use_camera=False,  # TODO
                         neg_reward=neg_reward,
                         body_name="hand_palm_link",
                         steps_per_action=10,
                         image_dimensions=None)

        left_finger_name = 'hand_l_distal_link'
        self._finger_names = [left_finger_name, left_finger_name.replace('_l_', '_r_')]
        self._set_new_goal()
        self._action_multiplier = action_multiplier
        self._continuous = continuous
        obs_shape = history_len * np.size(self._obs()) + np.size(self._goal())
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs_shape)

        if continuous:
            self.action_space = spaces.Box(-1, 1, shape=self.sim.nu)
        else:
            self.action_space = spaces.Discrete(self.sim.nu * 2 + 1)

        self.relation_manager = RelationManager()
        self.context_qpos = np.copy(self.sim.qpos)

        block1 = lambda: self.get_block1_position(self.context_qpos)
        block2 = lambda: self.get_block2_position(self.context_qpos)
        gripper = lambda: self._gripper_pos(self.context_qpos)

        touching = lambda o1, o2: self.are_positions_touching(o1, o2, touching_threshold=0.05)
        near = lambda o1, o2: self.are_positions_touching(o1, o2, touching_threshold=0.20)

        self.block1_idx = self.sim.jnt_qposadr('block1joint')
        self.block2_idx = self.sim.jnt_qposadr('block2joint')

        self.relation_manager.register_object('block1', block1)
        self.relation_manager.register_object('block2', block2)
        self.relation_manager.register_object('gripper', gripper)
        self.relation_manager.register_relationship('touching', touching, 2)
        self.relation_manager.register_relationship('near', near, 2)



    def generate_valid_block_position(self):
        low_range = np.array([-0.15, -0.25, 0.49])
        high_range = np.array([0.15, 0.25, 0.49])
        return np.random.uniform(low=low_range, high=high_range)

    def get_block1_position(self, qpos):
        position = qpos[self.block1_idx:self.block1_idx+3]
        return position

    def get_block2_position(self, qpos):
        position = qpos[self.block2_idx:self.block2_idx+3]
        return position

    def set_block_position(self, qpos, name, position):
        idx = self.sim.jnt_qposadr(name)
        qpos = np.copy(qpos)
        qpos[idx:idx+3] = position
        return qpos

    def are_positions_touching(self, pos1, pos2, touching_threshold=0.05):
        weighting = np.array([1, 1, 0.1])
        dist = np.sqrt(np.sum(weighting*np.square(pos1 - pos2)))
        return dist < touching_threshold


    def reset_qpos(self):
        qpos = self.init_qpos
        qpos = self.set_block_position(qpos, 'block1joint', self.generate_valid_block_position())
        qpos = self.set_block_position(qpos, 'block2joint', self.generate_valid_block_position())
        return qpos

    def _set_new_goal(self):
        goal_block = np.random.randint(0, 2)
        onehot = np.zeros([2],dtype=np.float32)
        onehot[goal_block] = 1
        self.__goal = onehot

    def _obs(self):
        return [self.sim.qpos]

    def _goal(self):
        return [self.__goal]

    def goal_3d(self):
        return [0,0,0]

    def _currently_failed(self):
        return False

    def at_goal_touch_block(self, qpos, goal):
        self.context_qpos = qpos
        goal_block = np.argmax(goal) + 1
        relations = self.relation_manager.compute_relations()
        if goal_block == 1:
            return ('block1', 'gripper') in relations['touching']
        else:
            return ('block2', 'gripper') in relations['touching']

    def at_goal_push_together(self, qpos, goal):
        self.context_qpos = qpos
        relations = self.relation_manager.compute_relations()
        return ('block1', 'block2') in relations['near']



    def at_goal(self, qpos, goal):
        #return False
        #return self.at_goal_touch_block(qpos, goal)
        return self.at_goal_push_together(qpos, goal)


    def _compute_terminal(self, goal, obs):
        goal, = goal
        qpos, = obs
        return self.at_goal(qpos, goal)

    def _compute_reward(self, goal, obs):
        qpos, = obs
        if self.at_goal(qpos, goal):
            return 1
        elif self._neg_reward:
            return -.0001
        else:
            return 0

    def _obs_to_goal(self, obs):
        raise Exception('No promises here.')
        qpos, = obs
        return [self._gripper_pos(qpos)]

    def _gripper_pos(self, qpos=None):
        finger1, finger2 = [self.sim.get_body_xpos(name, qpos)
                            for name in self._finger_names]
        return (finger1 + finger2) / 2.

    def step(self, action):
        if not self._continuous:
            ctrl = np.zeros(self.sim.nu)
            if action != 0:
                ctrl[(action - 1) // 2] = (1 if action % 2 else -1) * self._action_multiplier
            return BaseEnv.step(self, ctrl)
        else:
            action = np.clip(action * self._action_multiplier, -1, 1)
            return BaseEnv.step(self, action)


class RelationManager(object):

    def __init__(self):
        self.relationships = dict()
        self.objects = dict()
        self.relationship_tree = dict()

    def register_relationship(self, name, test, num_objects, assymmetric=False):
        if name in self.relationships:
            raise Exception('Relationship already exists with name %s' % name)
        self.relationships[name] = (test, num_objects)
        # add objects into relationship tree for easier computation
        if num_objects not in self.relationship_tree:
            self.relationship_tree[num_objects] = {}
        relationship_dict = self.relationship_tree[num_objects]
        relationship_dict[name] = test

        if assymmetric:
            if num_objects != 2:
                raise Exception('Assymmetry is only supported for relations with 2 objects.')
            flipped_test = lambda o1, o2: test(o2, o1)
            self.register_relationship(name+'__assymm', flipped_test, 2, assymmetric=False)

    def register_object(self, name, position_accessor):
        if name in self.objects:
            raise Exception('Object already exists with name %s' % name)
        self.objects[name] = position_accessor

    #def realize_objects(self):
    #    self.realized_objects = {name: accessor() for name, accessor in self.objects.items()}


    def compute_relations(self):
        output = {relation_name: [] for relation_name in self.relationships if not relation_name.endswith('__assymm')}
        realized_objects = {name: accessor() for name, accessor in self.objects.items()}
        for num_objects, relationship_dict in self.relationship_tree.items():
            # get pairs of objects and names
            for object_names in product(self.objects, repeat=num_objects):
                object_positions = [realized_objects[name] for name in object_names]
                for relation_name, relation in relationship_dict.items():
                    relation_value = relation(*object_positions)
                    if relation_value:
                        if relation_name.endswith('__assymm'):
                            output[relation_name.split('__assymm')[0]].append(tuple(object_names)[::-1])
                        else:
                            output[relation_name].append(tuple(object_names))
        return output





if __name__ == '__main__':
    import pprint
    pp = pprint.PrettyPrinter()

    near = lambda o1, o2: np.sqrt(np.sum(np.square(o1 - o2))) < 0.1
    far = lambda o1, o2: np.sqrt(np.sum(np.square(o1 - o2))) > 1.0
    behind = lambda o1, o2: np.all(np.less(o1, o2))

    def between(o1, o2, o3):
        # o1 is between o2 and o3 if it is sufficiently close to the line between o2 and o3
        x0, y0 = o1[0], o1[1]
        x1, y1 = o2[0], o2[1]
        x2, y2 = o3[0], o3[1]
        a = y2 - y1
        b = x1 - x2
        c = (x2 - x1)*y1 - (y2 - y1)*x1
        dist = np.abs(a*x0 + b*y0 + c) / np.sqrt(a**2 + b**2)
        return dist < 0.01


    object2 = lambda: np.array([0.0, 0.0])
    object1 = lambda: np.array([-1.0, -1.0])


    manager = RelationManager()
    manager.register_relationship('NEAR', near, 2)
    manager.register_relationship('FAR', far, 2)
    manager.register_relationship('BETWEEN', between, 3)
    manager.register_relationship('BEHIND', behind, 2, assymmetric=True)

    manager.register_object('OBJECT1', object1)
    manager.register_object('OBJECT2', object2)

    pp.pprint(manager.compute_relations())




