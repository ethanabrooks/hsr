import numpy as np



# object relationships
# NEAR, FAR, BETWEEN
class RelationshipManager(object):

    def __init__(self, close_threshold, far_threshold, between_threshold):
        self.registered_objects = {}
        self.close_threshold = close_threshold
        self.far_threshold = far_threshold
        self.between_threshold = between_threshold

    def register_object(self, name, position):
        if name in self.register_objects:
            raise Exception('%s is already registered.')
        self.register_objects[name] = position


    def distance(self, x, y):
        return np.sqrt(np.sum(np.square(x - y)))

    def compute_single_object_relationships(self, agent, object):
        dist = self.distance(object, agent)
        NEAR = dist < self.close_threshold
        FAR = dist > self.far_threshold
        return [NEAR, FAR]



    def compute_pair_object_relationships(self, agent, object1, object2):
        # compute between
        pass



    def get_relationships(self, agent):
        for name, object in self.registered_objects:
            if self.dist(object, agent) < self.close_threshold:
                pass


    def relationships_to_string(self, relationships):