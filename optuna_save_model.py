from tensorflow.keras.utils import plot_model

class Objective:
    def __init__(self, 
                method_class, 
                data_train, 
                data_test,
                path_params,
                data_val=None, 
                learning_scheduler=None,
                model_opts=None,
                callback=None):
        self.method_class = method_class
        self.data_train = data_train
        self.data_val = data_val
        self.data_test = data_test
        self.path_params = path_params
        self.learning_scheduler = learning_scheduler
        self.model_opts = model_opts
        self.callback = callback

    def __call__(self, trial):
        param = self.get_param(trial)
        model = self.method_class.trainer(trial, **param)
        self.callback.register_model(trial.number, model)
        acc = model.tester(self.data_test, model,  self.model_opts)

        return acc
        
    def get_param(self, trial):
        param = {
            'data_train': self.data_train,
            'data_val': self.data_val,
            'path_params': self.path_params,
            'learning_scheduler': self.learning_scheduler,
            'model_opts': self.model_opts
        }
        return param

class Callback:
    def __init__(self, model_path, model_img_path):
        self.model_path = model_path
        self.model_img_path = model_img_path
        self.models = {}

    def register_model(self, trial_number, model, hisotry):
        self.models[str(trial_number)] = {'model': model, 'history': hisotry}

    def unregister_model(self, trial_number):
        self.models.pop(str(trial_number), None)
    def unregister_other_model(self, trial_number):
        model = self.models.pop(str(trial_number), None)
        self.models.clear()
        self.models[str(trial_number)] = model

    def get_model(self, trial_number):
        return self.models[str(trial_number)]['model']
    def get_hisotry(self, trial_number):
        return self.models[str(trial_number)]['history']

    def __call__(self, study, trial):
        if study.best_trial.number == trial.number:
            self.unregister_other_model(study.best_trial.number)
            self.save_model(study)
        else:
            self.unregister_model(trial.number)
            # save study.trials_dataframe()
    def save_model(self, study):
        model = self.get_model(study.best_trial.number)
        print('Train model is saved to {}'.format(self.model_path))
        model.save(self.model_path)
        plot_model(model, to_file=self.model_img_path)
        print(study.best_params)

    def get_best_param(self, study):
        history = self.get_hisotry(study.best_trial.number)
        best_params = study.best_params
        best_params['history'] = history
        return best_params
        