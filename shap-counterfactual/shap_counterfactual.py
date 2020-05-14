"""
Function for explaining classified instances using evidence counterfactuals.
"""

"""
Import libraries 
"""
import shap
import time
import numpy as np
from scipy import sparse

class ShapCounterfactual(object):
    """Class for generating evidence counterfactuals for classifiers on behavioral/text data"""
    
    def __init__(self, classifier_fn, threshold_classifier, 
                 feature_names_full, max_features = 30,
                 time_maximum = 120):
        
        """ Init function
        
        Args:
            
            classifier_fn: [function] classifier prediction probability function
            or decision function. For ScikitClassifiers, this is classifier.predict_proba 
            or classifier.decision_function or classifier.predict_log_proba.
            Make sure the function only returns one (float) value. For instance, if you
            use a ScikitClassifier, transform the classifier.predict_proba as follows:
                
                def classifier_fn(X):
                    c = classification_model.predict_proba(X)
                    y_predicted_proba = c[:,1]
                    return y_predicted_proba
            
            max_features: [int] maximum number of features allowed in the explanation(s).
            Default is set to 30.
                                    
            threshold_classifier: [float] the threshold that is used for classifying 
            instances as positive or not. When score or probability exceeds the 
            threshold value, then the instance is predicted as positive. 
            We have no default value, because it is important the user decides 
            a good value for the threshold. 
            
            feature_names_full: [numpy.array] contains the interpretable feature names, 
            such as the words themselves in case of document classification or the names 
            of visited URLs. 
            (It can also be the indices if there are no interpretable feature names.)
            
            time_maximum: [int] maximum time allowed to generate explanations,
            expressed in minutes. Default is set to 2 minutes (120 seconds).
        """
        
        self.classifier_fn = classifier_fn
        self.max_features = max_features
        self.threshold_classifier = threshold_classifier
        self.feature_names_full = feature_names_full
        self.time_maximum = time_maximum 
    
    def explanation(self, instance):
        """ Generates evidence counterfactual explanation for the instance.
        
        Args:
            instance: [raw text string] instance to explain as a string
            with raw text in it
                        
        Returns:
            A dictionary where:
                
                explanation_set: features in counterfactual explanation.
                
                feature_coefficient_set: corresponding importance weights (SHAP) 
                of the features in counterfactual explanation.
                
                number_active_elements: number of active elements of 
                the instance of interest.
                                
                minimum_size_explanation: number of features in the explanation.
                
                minimum_size_explanation_rel: relative size of the explanation
                (size divided by number of active elements of the instance).
                
                time_elapsed: number of seconds passed to generate explanation.
                
                score[0]: predicted score/probability for instance.
                
                score_new[0]: predicted score/probability for instance when
                removing the features in the explanation set (~setting feature
                values to zero).
                
                difference_scores[0]: difference in predicted score/probability
                before and after removing features in the explanation.
                
                expl_shap: original explanation using SHAP (all active features
                with corresponding importance weights)
        """
        
        tic = time.time() #start timer
        
        #perturbing features means setting feature value to zero, so the reference values are zeroes
        reference = np.reshape(np.zeros(np.shape(instance)[1]), (1,len(np.zeros(np.shape(instance)[1]))))
        reference = sparse.csr_matrix(reference)

        explainer = shap.KernelExplainer(self.classifier_fn, reference, link="identity") 
        shap_values = explainer.shap_values(instance, nsamples=5000)
        
        nb_active_feature_instance_idx = np.size(instance)
        instance_dense = np.reshape(instance,(1,len(self.feature_names_full)))
        instance_dense = instance_dense.toarray() 
        
        explanation_shap = []
        indices_features_explanation_shap = []
        features_explanation_shap = []
        ind = 0
        output_size_shap = 0
        for i in instance_dense[0]:
            if (i != 0): #only for the active features of the instance
                explanation_shap.append(shap_values[:,ind])
                indices_features_explanation_shap.append(ind)
                features_explanation_shap.append(self.feature_names_full[ind])
                if (shap_values[:,ind]!=0):
                    output_size_shap += 1
            ind += 1
        if (output_size_shap == 0):
            output_size_shap = np.nan
        
        #Sort by decreasing absolute value 
        inds = np.argsort(np.array(np.abs(explanation_shap)), axis=0)
        inds = np.fliplr([inds])[0]
        
        indices_features_explanation_shap_abs = np.array(indices_features_explanation_shap)[inds]
        features_explanation_shap_abs = np.array(features_explanation_shap)[inds]
        explanation_shap_sorted = np.array(explanation_shap)[inds]
    
        length = 0
        iteration = 0
        score = self.classifier_fn(instance)
        score_new = score
        while ((score_new[0] >= self.threshold_classifier) and (length != len(explanation_shap_sorted)) and (length < self.max_features) and ((time.time()-tic) < self.time_maximum)):
            indices_features_explanations_shap_abs_found = []
            coefficients_features_explanations_shap_abs_found = []
            feature_names_full_index = []
            number_perturbed = 0
            length += 1
            perturbed_instance = instance.copy()
            j = 0
            for feature in indices_features_explanation_shap_abs[0:length]:
                if (explanation_shap_sorted[j] >= 0):
                    perturbed_instance[:,feature] = 0
                    number_perturbed += 1
                    indices_features_explanations_shap_abs_found.append(feature[0])
                    feature_names_full_index.append(features_explanation_shap_abs[j])
                    coefficients_features_explanations_shap_abs_found.append(explanation_shap_sorted[j][0][0])
                j += 1
            score_new = self.classifier_fn(perturbed_instance)
            iteration += 1
            
        if (score_new[0] < self.threshold_classifier):            
            time_elapsed = time.time() - tic
            minimum_size_explanation = number_perturbed
            minimum_size_explanation_rel = number_perturbed/nb_active_feature_instance_idx
            difference_scores = score - score_new
            number_active_elements = nb_active_feature_instance_idx
            expl_shap = explanation_shap_sorted
            explanation_set = feature_names_full_index[0:number_perturbed]
            feature_coefficient_set = coefficients_features_explanations_shap_abs_found[0:number_perturbed]
            
        else:
            time_elapsed = np.nan
            minimum_size_explanation = np.nan
            minimum_size_explanation_rel = np.nan
            difference_scores = np.nan
            number_active_elements = nb_active_feature_instance_idx
            expl_shap = np.nan
            explanation_set = []
            feature_coefficient_set = []            
            
        return {'explanation_set':explanation_set, 'feature_coefficient_set':feature_coefficient_set, 'number_active_elements':number_active_elements, 'size explanation': minimum_size_explanation, 'relative size explanation':minimum_size_explanation_rel, 'time elapsed':time_elapsed, 'original score':score[0], 'new score':score_new[0], 'difference scores':difference_scores[0], 'explanation SHAP coefficients':expl_shap}      