from FuncionesAux import *
k = make_Kpods()
lista_clientes, lastCommits = create_clients(k)

print("Adelante party "+ str(1))
dir_base = os.getcwd()
(x_train, y_train), (x_test, y_test), n_features, model = read_client(lista_clientes= lista_clientes, lastCommits=lastCommits,
                                                                          client=10)
print("Fit part")

model, pred = FitForecast_e(model, x_train, y_train, x_test, 24, 7,
                                                 24, n_features, 10, None)