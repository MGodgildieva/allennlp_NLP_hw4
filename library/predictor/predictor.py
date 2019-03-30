from allennlp.predictors import Predictor
from allennlp.common.util import JsonDict
import numpy as np

@Predictor.register('name-predictor')
class LngPredictor(Predictor):
    
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._dataset_reader.text_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        tag_ids = np.argmax(output_dict['logits'], axis=-1)
        return [self._model.vocab.get_token_from_index(i, 'labels') for i in tag_ids]
