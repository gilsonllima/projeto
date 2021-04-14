# codigo para captura de movimento espacial de um corpo e calculo do desvio padrao

from imutils import face_utils, resize
import dlib
import cv2
import numpy as np
import pandas as pd
import os

# funcao para retornar coordenadas dos marcadores do dlib

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)  # inicializa lista de coordenadas (x, y)
    for i in range(0, 68):  # loop sobre os 68 marcadores faciais / conversao em tuplas de coordenadas (x, Y)
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords  # retorna lista de coordenadas (x, y)

# arquivo a ser analisado

arquivo = "caminho arquivo"

# p = modelo pre treinado / esta na pasta de trabalho. disponivel em http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# inicia lista de angulos de Euler

lista_angulos = []

# inicia captura de video

cap = cv2.VideoCapture(arquivo)

while True:

    # captura do frame

    _, image = cap.read()

    # variavel utilizada para configuracao da camera

    size = image.shape

    # teste de dimensoes do video / diminuir para otimizar processamento

    if size[0] > 640 or size[1] > 640:
        escala = 35  # percentual do tamanho original
        largura = int(image.shape[1] * escala / 100)
        altura = int(image.shape[0] * escala / 100)
        dim = (largura, altura)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        size = image.shape

    # converter para escala de cinza / otimizar processamento

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detectando a(s) face(s)

    rects = detector(gray, 0)

    # para cada face encontrada, encontrar marcadores faciais

    for (i, rect) in enumerate(rects):
        
        # executa a predicao e a converte em numpy array
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # pontos 2D da imagem analisada em locais especificos para estimar movimento no espaco

        image_points = np.array([
            (shape[30][0], shape[30][1]),  # Ponta nariz
            (shape[8][0], shape[8][1]),  # Queixo
            (shape[36][0], shape[36][1]),  # Canto externo olho esquerdo
            (shape[45][0], shape[45][1]),  # Canto externo olho direito
            (shape[48][0], shape[48][1]),  # Canto esquerdo boca
            (shape[54][0], shape[54][1])  # Canto direito boca
        ], dtype="double")

        # pontos modelo generico 3D de referencia

        model_points = np.array([
            (0.0, 0.0, 0.0),  # Ponta nariz
            (0.0, -330.0, -65.0),  # Queixo
            (-225.0, 170.0, -135.0),  # Canto externo olho esquerdo
            (225.0, 170.0, -135.0),  # Canto externo olho direito
            (-150.0, -150.0, -125.0),  # Canto esquerdo boca
            (150.0, -150.0, -125.0)  # Canto direito boca
        ])

        # caracteristicas da camera / modelo generico

        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        # assumindo que nao ha distorcao da lente

        dist_coeffs = np.zeros((4, 1))
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        # mostrar pontos de referencia espacial

        for p in image_points:
            cv2.circle(image, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        # desenhar linhas de referencia para facilitar compreensao 3D

        cv2.line(image, (int(image_points[0][0]), int(image_points[0][1])),
                 (int(image_points[2][0]), int(image_points[2][1])), (255, 51, 255), 2)
        cv2.line(image, (int(image_points[0][0]), int(image_points[0][1])),
                 (int(image_points[3][0]), int(image_points[3][1])), (255, 51, 255), 2)
        cv2.line(image, (int(image_points[2][0]), int(image_points[2][1])),
                 (int(image_points[3][0]), int(image_points[3][1])), (255, 51, 255), 2)
        cv2.line(image, (int(image_points[2][0]), int(image_points[2][1])),
                 (int(image_points[4][0]), int(image_points[4][1])), (255, 51, 255), 2)
        cv2.line(image, (int(image_points[3][0]), int(image_points[3][1])),
                 (int(image_points[5][0]), int(image_points[5][1])), (255, 51, 255), 2)
        cv2.line(image, (int(image_points[4][0]), int(image_points[4][1])),
                 (int(image_points[1][0]), int(image_points[1][1])), (255, 51, 255), 2)
        cv2.line(image, (int(image_points[5][0]), int(image_points[5][1])),
                 (int(image_points[1][0]), int(image_points[1][1])), (255, 51, 255), 2)
        cv2.line(image, (int(image_points[4][0]), int(image_points[4][1])),
                 (int(image_points[5][0]), int(image_points[5][1])), (255, 51, 255), 2)
        cv2.line(image, (int(image_points[4][0]), int(image_points[4][1])),
                 (int(image_points[0][0]), int(image_points[0][1])), (255, 51, 255), 2)
        cv2.line(image, (int(image_points[5][0]), int(image_points[5][1])),
                 (int(image_points[0][0]), int(image_points[0][1])), (255, 51, 255), 2)
        cv2.line(image, (int(image_points[0][0]), int(image_points[0][1])),
                 (int(image_points[1][0]), int(image_points[1][1])), (255, 51, 255), 2)

        # Referencias para construcao dos eixos na tela

        (origem, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 0.0)]), rotation_vector,
                                               translation_vector, camera_matrix, dist_coeffs)
        (eixox, jacobian) = cv2.projectPoints(np.array([(500, 0.0, 0.0)]), rotation_vector,
                                              translation_vector, camera_matrix, dist_coeffs)
        (eixoy, jacobian) = cv2.projectPoints(np.array([(0.0, 500, 0.0)]), rotation_vector,
                                              translation_vector, camera_matrix, dist_coeffs)
        (eixoz, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector,
                                              translation_vector, camera_matrix, dist_coeffs)

        # Pontos de referencia para contrucao dos eixos

        oo = (int(image_points[0][0]), int(image_points[0][1]))
        ox = (int(eixox[0][0][0]), int(eixox[0][0][1]))
        oy = (int(eixoy[0][0][0]), int(eixoy[0][0][1]))
        oz = (int(eixoz[0][0][0]), int(eixoz[0][0][1]))

        # procedimento para obter angulos de Euler

        rmat, jac = cv2.Rodrigues(rotation_vector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        (x, y, z) = angles

        # adequação matematica para melhor compreensao do dado

        if x == 180:
            x = 0
        else:
            if x > 0:

                x = (-1) * (x - 180)
            else:
                x = (-1) * (x + 180)

        # exibir angulos de Euler na tela

        cv2.putText(image, "Angulos Euler:", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, "Pitch: {}".format(round(x, 2)), (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, "Yaw: {}".format(round(y, 2)), (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, "Roll: {}".format(round(z, 2)), (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # exibir os eixos xyz fixo e movel para faciliar vizualizacao
        # atenção a sobreposicao das linhas de maneira a reforcar impressao de tridimensionalidade

        if x < 0:
            cv2.line(image, oo, (oo[0], oo[1] - 85), (0, 255, 0), 2)
            cv2.line(image, oo, oy, (0, 130, 0), 2)
        if x >= 0:
            cv2.line(image, oo, oy, (0, 130, 0), 2)
            cv2.line(image, oo, (oo[0], oo[1] - 85), (0, 255, 0), 2)

        if y < 0:
            cv2.line(image, oo, ox, (0, 0, 139), 2)
            cv2.line(image, oo, (oo[0] + 85, oo[1]), (0, 0, 255), 2)
        if y >= 0:
            cv2.line(image, oo, (oo[0] + 85, oo[1]), (0, 0, 255), 2)
            cv2.line(image, oo, ox, (0, 0, 139), 2)

        cv2.line(image, oo, oz, (130, 0, 0), 2)

        # nota: o eixo z de referencia fica perpendicular a tela e eh representado aqui como um ponto

        cv2.circle(image, (int(oo[0]), int(oo[1])), 5, (255, 0, 0), -1)
        cv2.circle(image, (int(oo[0]), int(oo[1])), 2, (0, 0, 0), -1)

        # lista para concatenar e adicionar ao dataframe

        angulos = (x, y, z)

    # exibir imagem

    cv2.imshow("Output", image)

    # inclui novas observacoes na lista

    lista_angulos.append(angulos)

    # condicao de saida do loop <ESC>

    k = cv2.waitKey(33) & 0xFF
    if k == 27:
        break

# criacao de dataframe com todas as observações de pitch, yaw e rool

df = pd.DataFrame(lista_angulos, columns=["pitch", "yaw", "roll"])

# desvio padrao das obervacoes

print("\nDados extraidos do arquivo:\n\n{}\n".format(arquivo.title().upper()))
print("Desvio padrão pitch: {} ".format(round(np.std(df.pitch), 2)))
print("Desvio padrão yaw: {} ".format(round(np.std(df.yaw), 2)))
print("Desvio padrão roll: {} ".format(round(np.std(df.roll), 2)))

# criacao de dataframe com desvios padroes

df_sd = pd.DataFrame({"pitch": [np.std(df.pitch)],
                      "yaw": [np.std(df.yaw)],
                      "roll": [np.std(df.roll)]},
                     index=["sd"])

# adiciona a linha com o valor de cada desvio padrao

df = pd.concat([df, df_sd])

# contruir nome do arquivo CSV de saida

nome_arquivo = arquivo.title().upper().split(sep=".")
nome = str(nome_arquivo[0])
arq_ext = str(nome_arquivo[1].lower())

# escrever arquivo CSV com observacoes

df.to_csv(nome + "." + arq_ext + '.csv')

print("\nArquivo CSV salvo em {}".format(os.path.dirname(arquivo)))

# fechar janelas e terminar tarefa

cv2.destroyAllWindows()
cap.release()
