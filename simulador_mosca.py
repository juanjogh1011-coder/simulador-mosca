import os
import sys
import numpy as np
from dotenv import load_dotenv
from caveclient import CAVEclient
import brian2 as b2
from dm_control import mujoco
from flybody import fly_envs

def get_flywire_client():
    load_dotenv()
    mi_token = os.getenv('FLYWIRE_TOKEN')
    if mi_token:
        print("Configurando token desde .env...")
        temp_client = CAVEclient(server_address="https://global.daf-apis.com")
        temp_client.auth.save_token(token=mi_token, overwrite=True)

    try:
        client = CAVEclient('flywire_fafb_production')
        # Check if auth works
        client.materialize.get_tables()
        print("Autenticación con FlyWire exitosa.")
    except Exception as e:
        print(f"Error de autenticación: {e}")
        sys.exit(1)
    return client

def setup_brian2_network():
    b2.prefs.codegen.target = 'numpy'
    b2.start_scope()
    
    # Modelo Leaky Integrate-and-Fire
    tau = 10*b2.ms
    v_rest = -65*b2.mV
    v_th = -50*b2.mV
    v_reset = -65*b2.mV
    R = 10*b2.Mohm
    
    eqs = '''
    dv/dt = -(v - v_rest)/tau + R*I/tau : volt
    I : amp
    '''
    
    sensory_group = b2.NeuronGroup(2, eqs, threshold='v > v_th', reset='v = v_reset', method='exact')
    sensory_group.v = v_rest
    sensory_group.I = 0*b2.nA
    
    motor_group = b2.NeuronGroup(2, eqs, threshold='v > v_th', reset='v = v_reset', method='exact')
    motor_group.v = v_rest
    motor_group.I = 0*b2.nA
    
    # Conexiones sinápticas excitatorias simples
    S = b2.Synapses(sensory_group, motor_group, on_pre='v_post += 5*mV')
    S.connect(i=0, j=0)
    S.connect(i=1, j=1)
    
    # Monitores
    sensory_spikes = b2.SpikeMonitor(sensory_group)
    motor_spikes = b2.SpikeMonitor(motor_group)
    
    net = b2.Network(sensory_group, motor_group, S, sensory_spikes, motor_spikes)
    
    return net, sensory_group, motor_group, sensory_spikes, motor_spikes

def main():
    # 1. Autenticación FlyWire
    print("Inicializando cliente de FlyWire...")
    client = get_flywire_client()
    
    # 2. Red Neuronal (Brian2)
    print("Configurando red neuronal biológica...")
    net, sensory_group, motor_group, sensory_spikes, motor_spikes = setup_brian2_network()
    
    # 3. Entorno Físico (FlyBody/MuJoCo)
    print("Inicializando entorno físico de FlyBody...")
    env = fly_envs.walk_on_ball()
    action_spec = env.action_spec()
    timestep = env.reset()
    
    # 4. Bucle de Control en Tiempo Real
    print("Iniciando simulación...")
    num_steps = 50
    dt_ms = 10 # 10 ms por iteración
    
    # Preparar acción nula
    action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
    
    prev_counts = np.zeros(2)
    
    for step in range(num_steps):
        # Extraer ángulos de articulaciones (joints) de MuJoCo
        obs = None
        for k, v in timestep.observation.items():
            if isinstance(v, np.ndarray) and v.size >= 2:
                obs = v.flatten()[:2]
                break
        
        if obs is None:
            obs = np.array([0.0, 0.0])
            
        # Convertir ángulos a corriente (nA) inyectada
        currents = np.abs(obs) * 2
        sensory_group.I = currents * b2.nA
        
        # Correr simulación 10ms
        net.run(dt_ms * b2.ms)
        
        # Traducir spikes motores en torques/fuerzas
        current_counts = np.array(motor_spikes.count)
        diff_counts = current_counts - prev_counts
        prev_counts = current_counts.copy()
        
        action.fill(0)
        action[0] = diff_counts[0] * 0.1
        if action.size > 1:
            action[1] = diff_counts[1] * 0.1
            
        # Aplicar al modelo
        timestep = env.step(action)
        
        # Mostrar métricas cada 10 iteraciones
        if (step + 1) % 10 == 0:
            print(f"--- Iteración {step + 1} ---")
            print(f"Corriente Inyectada: {currents} nA")
            print(f"Spikes Sensoriales Totales: {np.array(sensory_spikes.count)}")
            print(f"Spikes Motores Totales: {np.array(motor_spikes.count)}")
            print(f"Acción (Torques) aplicada: {action[:2]}...")

if __name__ == '__main__':
    main()
