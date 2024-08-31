setenv('KMP_DUPLICATE_LIB_OK', 'TRUE');
Predictor = py.importlib.import_module('Predictor');
% disp(Predictor)
ModelRunner = Predictor.ModelRunner();
% disp(ModelRunner)
disp(ModelRunner.model);
ModelRunner.load_model();
disp(ModelRunner.model);
% deepON_module = py.importlib.import_module('DeepONet');
% DeepON_class = deepON_module.DeepON;
% disp(DeepON_class);
