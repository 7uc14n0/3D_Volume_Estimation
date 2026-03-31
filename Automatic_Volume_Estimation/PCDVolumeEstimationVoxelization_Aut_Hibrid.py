## Este código é um exemplo de como acumular pontos de um LiDAR em tempo real e exibi-los em uma janela 3D usando Open3D.
# em seguida o código idenifica o objeto a ser calculado o volume, segnmenta o objeto e calcula o volume. 
# Ao final, o código exibe a nuvem de pontos acumulada, a caixa delimitadora do objeto identificado, as medidas do objeto 
# e o volume estimado, além disso, exibe o tempo de execução e salva a nuvem acumulada em um arquivo PCD.
# 
import sys
import time
import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import struct
import scipy.spatial
from pathlib import Path    # Importando Path da biblioteca pathlib para manipulação de caminhos de arquivos
import csv  # Importando a biblioteca csv para manipulação de arquivos CSV
import os # Importando a biblioteca os para manipulação de arquivos e diretórios
from sklearn.cluster import DBSCAN  # Importando DBSCAN do scikit-learn para segmentação de nuvens de pontos

start_time = 0   # Iniciando o cronômetro para medir o tempo de execução
# Variáveis globais para controle de tentativas e sucesso
tests = 0 # Contador de testes de coleta de dados do LiDAR
testsOk = 0 # Contador de testes bem-sucedidos
Volume_hist = []  # Lista para armazenar volumes calculados
endReadSensor = False  # Variável para controlar o término da leitura do sensor

start_time = time.time()
class LidarSubscriber(Node):
    def __init__(self):
        super().__init__('python_node')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/unilidar/cloud',
            self.callback,
            10
        )
        self.subscription

        # Definindo os limites do espaço 3D
        self.z_min = 0.15
        self.z_max = 1.5
        self.x_min = -0.25
        self.x_max = 2.0
        self.y_min = -1.5
        self.y_max = 1.5
        self.accumulated_points = []

        # Volume padrão para comparação
        self.volumeStandard = 0.06602  # Volume padrão em m³ (ajustável conforme necessário)

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(f"Nuvem de pontos Tentativa {tests}", width=800, height=600)
        self.point_cloud = o3d.geometry.PointCloud()
        self.geometry_added = False
        self.vis.get_render_option().point_size = 0.5 # Tamanho do ponto na visualização

    def callback(self, msg):
        new_points = self.convert_pointcloud2_to_numpy(msg)
        new_points = new_points[(new_points[:, 0] >= self.x_min) & (new_points[:, 0] <= self.x_max) &
                                (new_points[:, 1] >= self.y_min) & (new_points[:, 1] <= self.y_max) &
                                (new_points[:, 2] >= self.z_min) & (new_points[:, 2] <= self.z_max)]
        self.accumulated_points.extend(new_points)

        self.point_cloud.points = o3d.utility.Vector3dVector(np.array(self.accumulated_points))
        if not self.geometry_added:
            self.vis.add_geometry(self.point_cloud)
            self.geometry_added = True
        else:
            self.vis.update_geometry(self.point_cloud)
        self.vis.poll_events()
        self.vis.update_renderer()

    def convert_pointcloud2_to_numpy(self, msg):
        point_step = msg.point_step
        data = msg.data
        points = []
        for i in range(0, len(data), point_step):
            try:
                x, y, z = struct.unpack_from('fff', data, i)
                points.append([x, y, z])
            except struct.error:
                continue
        return np.array(points)
    
    def detect_and_segment_object(self):       
       # Função para verificar se o volume estabilizou
       def volume_estabilizou(volume_atual, volume_hist):
            """
            Verifica se o volume estimado estabilizou com base na variação das últimas leituras.

            Parâmetros:
            - volume_atual: float - volume calculado na iteração atual.
            - volume_hist: list[float] - lista com os últimos volumes registrados.
            - janela: int - número de volumes consecutivos a considerar para verificação.
            - delta_thresh: float - variação máxima permitida entre volumes consecutivos (em m³).

            Retorno:
            - (bool): True se o volume estabilizou, False caso contrário.
            """
            janela = 3  # Número de volumes consecutivos a considerar
            delta_thresh = 0.00025  # mais rápido 0,00025 para caixas retangulares, mais preciso 0,000025
            # Adiciona o volume atual ao histórico
            volume_hist.append(volume_atual)

            # Mantém o histórico limitado ao tamanho da janela
            if len(volume_hist) > janela:
                volume_hist.pop(0)

            # Se ainda não temos histórico suficiente, não estabilizou
            if len(volume_hist) < janela:
                return False

            # Verifica as variações entre volumes consecutivos
            variacoes = [abs(volume_hist[i] - volume_hist[i - 1]) for i in range(1, janela)]
            
            # Se todas as variações estão dentro do limite, consideramos estabilizado
            if all(v < delta_thresh for v in variacoes):
                return True

            return False

       # Função para calcular volume via Voxelization
       def calcular_volume_hibrido(pcd, voxel_size=0.03, eps_z=0.03, eps_x=0.03, min_voxels=30):
            """
            Estima o volume de objetos irregulares combinando projeções XY (topos) e YZ (faces),
            e tirando a média dos volumes estimados para melhorar a precisão.

            Parâmetros:
            - pcd: open3d.geometry.PointCloud - nuvem de pontos segmentada.
            - voxel_size: float - tamanho dos voxels. (default 0.02)
            - eps_z: float - agrupamento de topos no eixo Z. (default 0.03)
            - eps_x: float - agrupamento de faces no eixo X. (default 0.03)
            - min_voxels: int - mínimo de voxels por grupo. (default 30)

            Retorna:
            - volume_total: float - volume estimado médio.
            - volume_topo: float - volume a partir dos topos (XY).
            - volume_face: float - volume a partir das faces (YZ).
            """

            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
            voxels = voxel_grid.get_voxels()
            if len(voxels) == 0:
                print("Nenhum voxel encontrado.")
                return 0.0, 0.0, 0.0

            voxel_centers_idx = np.array([v.grid_index for v in voxels])
            voxel_centers_real = voxel_centers_idx * voxel_size

            z_chao = np.max(voxel_centers_real[:, 2])  # maior Z (mais próximo do chão)
            x_fundo = np.max(voxel_centers_real[:, 0])  # maior X (profundidade)

            volume_topo = 0.0
            volume_face = 0.0

            # --- TOPO: projeção XY + altura relativa ---
            z_values = voxel_centers_real[:, 2].reshape(-1, 1) 
            db_topo = DBSCAN(eps=eps_z, min_samples=min_voxels).fit(z_values)
            for label in set(db_topo.labels_):
                if label == -1:
                    continue
                cluster = voxel_centers_real[db_topo.labels_ == label]
                xy_cells = set((int(v[0]/voxel_size), int(v[1]/voxel_size)) for v in cluster)
                area_xy = len(xy_cells) * voxel_size**2
                z_medio = np.mean(cluster[:, 2])
                altura = z_chao - z_medio
                volume_topo += area_xy * altura

            # --- FACE: projeção YZ + profundidade relativa ---
            x_values = voxel_centers_real[:, 0].reshape(-1, 1)
            db_face = DBSCAN(eps=eps_x, min_samples=min_voxels).fit(x_values)
            for label in set(db_face.labels_):
                if label == -1:
                    continue
                cluster = voxel_centers_real[db_face.labels_ == label]
                yz_cells = set((int(v[1]/voxel_size), int(v[2]/voxel_size)) for v in cluster)
                area_yz = len(yz_cells) * voxel_size**2
                x_medio = np.mean(cluster[:, 0])
                profundidade = x_fundo - x_medio
                volume_face += area_yz * profundidade

            # Volume médio entre as duas estimativas
            volume_total = (volume_topo + volume_face) / 2

            print(f"[TOPO] Volume estimado: {volume_topo:.6f} m³")
            print(f"[FACE] Volume estimado: {volume_face:.6f} m³")
            print(f"[MÉDIA] Volume final estimado: {volume_total:.6f} m³")

            return voxel_grid, volume_total

       #------------------------------------------------------------------------------------
       # Iniciar detecção e segmentação do objeto
       global testsOk, start_time, Volume_hist, endReadSensor  # Indica que será usado a variável global

       print("Iniciando detecção do objeto completo com Voxelization...")
       # Criar nuvem de pontos
       pcd = o3d.geometry.PointCloud()
       pcd.points = o3d.utility.Vector3dVector(np.array(self.accumulated_points))

       # Estimar normais da nuvem de pontos
       pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
       pcd.orient_normals_consistent_tangent_plane(k=30)

       # DBSCAN para segmentação em clusters
       labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=30, print_progress=False))
       max_label = labels.max()
       print(f"{max_label + 1} clusters encontrados.")

       if max_label < 0:
           print("Nenhum cluster encontrado.")
           return

       clusters = []
       centers = []

       for i in range(max_label + 1):
           indices = np.where(labels == i)[0]
           cluster_points = np.asarray(pcd.points)[indices]
           if len(cluster_points) >= 500: #500 # tamanho mínimo do cluster
               clusters.append(cluster_points)
               centers.append(np.mean(cluster_points, axis=0))

       if not clusters:
           print("Nenhum cluster com tamanho suficiente.")
           return

       # Encontrar o maior cluster
       cluster_sizes = [len(c) for c in clusters]
       largest_idx = np.argmax(cluster_sizes)
       largest_center = centers[largest_idx]

       # Combinar clusters próximos do maior
       merged_points = clusters[largest_idx]
       for i, cluster in enumerate(clusters):
           if i == largest_idx:
               continue
           dist = np.linalg.norm(np.array(centers[i]) - np.array(largest_center))
           if dist < 0.2:  # tolerância de proximidade (ajustável)
               merged_points = np.vstack((merged_points, cluster))

       print(f"Objeto identificado com {merged_points.shape[0]} pontos.")

       # Finalizar tempo de execução
       end_time = time.time()
       elapsed_time = end_time - start_time
       print(f"Tempo total de execução: {elapsed_time:.2f} segundos")
    
       # Após obter merged_points (substituir a parte da OBB):
       object_pcd = o3d.geometry.PointCloud()
       object_pcd.points = o3d.utility.Vector3dVector(merged_points)
        
       # Calcular volume com voxelização
       voxel_grid, volume = calcular_volume_hibrido(object_pcd)

       if volume_estabilizou(volume, Volume_hist):
            print("Volume estabilizado. Prosseguindo com o cálculo e salvamento.")
            endReadSensor = True
       # Salvar nuvem de pontos acumulada
       if endReadSensor:
            print("Tentativa bem-sucedida! Volume dentro do padrão.")
            testsOk += 1  # incrementa o contador
            # Criação do diretório (só precisa ser feito uma vez)
            output_dir = Path.home() / "PCDs/Voxelization/OutrosTestesRealTime/Palestra"  # Define o diretório de saída
            output_dir.mkdir(parents=True, exist_ok=True)

            # Adiciona o número da tentativa nos nomes dos arquivos
            num = testsOk  # Número da tentativa bem-sucedida
            # Caminhos para salvar os arquivos
            pcd_path = output_dir / f"nuvem_acumulada_{num}.pcd"
            #obb_path = output_dir / f"obb_{num}.ply"
            csv_path = output_dir / f"resultado_volume.csv"
            #log_path = output_dir / f"resultado_volume_{num}.txt"  # <-- log

            # Salvar nuvem de pontos e OBB
            o3d.io.write_point_cloud(str(pcd_path), object_pcd)
            print(f"Nuvem de pontos acumulada salva em '{pcd_path}'.")

            # Dados a salvar
            dados = {
                        "Número da tentativa": num,
                        "Número de pontos": len(object_pcd.points),
                        "Volume calculado (m³)": round(volume, 6),
                        "Tempo de execução": round(elapsed_time, 2)
                    }

            # Salvar CSV individual para cada tentativa
            # Verifica se o arquivo já existe
            file_exists = os.path.exists(csv_path)

            with open(csv_path, mode='a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=dados.keys())
                
                if not file_exists:
                    writer.writeheader()  # Escreve cabeçalho só na primeira vez
                
                writer.writerow(dados)   # Adiciona nova linha com os dados
            print(f"Resultados salvos em: {csv_path}")

            # Visualização
            sensor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            o3d.visualization.draw_geometries(
                    [object_pcd, voxel_grid, sensor_frame],
                    window_name=f"Objeto com Voxelização tentativa {testsOk}",
                    width=800,
                    height=600
                )


def main():

    global tests, testsOk, start_time, Volume_hist, endReadSensor  # Indica que será usado as variáveis globais
    tmax = 60  # Tempo de aquisição: 2 melhor (mais preciso), 0.3 s (default) 90 s(capta a caixa toda)
    # Inicio das tentativas de coletas de dados do LiDAR
    print("Iniciando coleta de dados do LiDAR...")
    # Loop de coleta de dados
    while tests < 10:  # Tenta até encontrar 10 tentativas bem-sucedidas
        rclpy.init()
        node = LidarSubscriber()
        start_time = time.time()    # Reinicia o cronômetro para a próxima tentativa
        while ((time.time() - start_time) < tmax) and not endReadSensor:
            rclpy.spin_once(node, timeout_sec=0.1)
            # Após tempo de coleta, detectar e segmentar o objeto
            node.detect_and_segment_object()
        tests += 1
        print(f"Tentativa {tests} concluída.")
        node.destroy_node()
        rclpy.shutdown()
        time.sleep(1)  # Pausa de 1 segundo entre as tentativas
        endReadSensor = False
        # Verifica se foram feitas 8 tentativas bem-sucedidas
        if testsOk == 8:
            print("Foram registradas 8 tentativas bem-sucedidas. Encerrando o programa.")
            break

if __name__ == '__main__':
    main()

