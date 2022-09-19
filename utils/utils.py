import math
from itertools import chain, combinations
from pathlib import Path
import numpy as np

class ConceptProperties:

    def calc_completeness_given_concept(self, predictor, decoder, latent, concepts, y_pred, used_concepts_index):
        """
        excluding concept, by setting picking closest other concept
        """
        concept_index_not_excluded = self.get_concept_not_excluded(concepts, used_concepts_index)
        # map each instance to concepts of subsets
        instance_concept = self.get_closest_concept_to_instances(latent, concepts[concept_index_not_excluded])
        if decoder:
            instance_concept = decoder(instance_concept)
        y_pred_concepts = predictor(instance_concept)

        return self.get_completness(y_pred, y_pred_concepts)


    @staticmethod
    def get_completness(y_classifier, y_explainer):
        completeness = sum(np.argmax(y_classifier, axis=1) == np.argmax(y_explainer, axis=1)) \
                       / len(np.argmax(y_explainer, axis=1))
        return completeness

    @staticmethod
    def get_closest_concept_to_instances(latent, centers):
        """
        returns closes concept for each instance
        """
        instance_concept = np.zeros_like(latent)
        for i, instance in enumerate(latent):
            distance = np.linalg.norm(instance - centers, axis=-1)
            distance_min_index = np.argsort(distance)[:1]
            instance_concept[i] = centers[distance_min_index]#
        return instance_concept

    @staticmethod
    def get_concept_not_excluded(concepts, used_concepts_index):
        """
        function replaces concept with nearest neighbour, returns new index for each concept
        """
        neighbour_index = np.zeros(len(concepts))
        for i, concept in enumerate(concepts):
            distance = np.linalg.norm(concept - concepts[used_concepts_index], axis=-1)
            neighbour_index[i] = np.argsort(distance)[:1]
        concept_index_not_excluded = [used_concepts_index[int(i)] for i in neighbour_index]
        return concept_index_not_excluded

    def get_concept_shap(self, predictor, decoder, latent, concepts, y_pred):
        """
        function initially from https://github.com/chihkuanyeh/concept_exp
        note for n_concepts > 10, this is very inefficient to calculate
        """
        conceptSHAP = []
        c_id = np.asarray(list(range(len(concepts))))

        print("start concept shap")
        for idx in c_id:

            exclude = np.delete(c_id, idx)
            subsets = np.asarray(self.powerset(list(exclude)))
            sum = 0
            for subset in subsets:
                # score 1:
                c1 = subset + [idx]
                score1 = self.calc_completeness_given_concept(predictor, decoder, latent, concepts, y_pred, c1)

                # score 2:
                c1 = subset
                if c1 != []:
                    score2 = self.calc_completeness_given_concept(predictor, decoder, latent, concepts, y_pred, c1)
                else:
                    score2 = 0

                norm = (math.factorial(len(c_id) - len(subset) - 1) * math.factorial(len(subset))) / \
                       math.factorial(len(c_id))
                sum += norm * (score1 - score2)
            conceptSHAP.append(sum)
        return conceptSHAP

    @staticmethod
    def powerset(iterable):
        """
        function from https://github.com/chihkuanyeh/concept_exp
        powerset([1,2,3]) --> [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]
        """
        s = list(iterable)
        pset = chain.from_iterable(combinations(s, r) for r in range(0, len(s) + 1))
        return [list(i) for i in list(pset)]
