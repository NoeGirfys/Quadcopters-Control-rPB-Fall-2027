"""
test_natnet.py — Diagnostic de connexion OptiTrack via NatNet SDK officiel
===========================================================================

Prérequis
---------
1. Télécharger le NatNet SDK depuis :
   https://www.optitrack.com/support/downloads/developer-tools.html

2. Copier le fichier NatNetClient.py (situé dans SDK/Samples/PythonClient/)
   dans le même dossier que ce script.

3. Lancer ce script :
   python test_natnet.py

Le script va :
  - Se connecter à Motive en multicast ET en unicast (teste les deux)
  - Afficher les rigid bodies reçus avec leur position
  - Diagnostiquer les problèmes de réseau/firewall
  - S'arrêter proprement après 10 secondes

Configuration
-------------
Modifier SERVER_IP et CLIENT_IP ci-dessous selon votre réseau.
"""

import sys
import os
import time
import socket
import threading

# Cherche NatNetClient.py d'abord dans le dossier courant,
# puis dans un sous-dossier 'NatNetSDK/' s'il existe
_sdk_subdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NatNetSDK")
if os.path.isdir(_sdk_subdir):
    sys.path.insert(0, _sdk_subdir)

# ─── CONFIGURATION ────────────────────────────────────────────────
SERVER_IP  = "192.168.0.24"   # IP de la machine Motive (OptiTrack PC)
CLIENT_IP  = "0.0.0.0"        # IP de ce PC (0.0.0.0 = auto-detect)
DURATION_S = 10               # Durée d'écoute en secondes
# ──────────────────────────────────────────────────────────────────


def check_natnet_client_available():
    """Vérifie que NatNetClient.py est présent dans le dossier."""
    try:
        import NatNetClient
        return NatNetClient
    except ImportError:
        print("=" * 60)
        print("ERREUR : NatNetClient.py introuvable !")
        print()
        print("Étapes pour l'installer :")
        print("  1. Télécharger le NatNet SDK depuis :")
        print("     https://www.optitrack.com/support/downloads/developer-tools.html")
        print("  2. Extraire l'archive")
        print("  3. Copier le fichier :")
        print("     SDK/Samples/PythonClient/NatNetClient.py")
        print("     dans le même dossier que ce script")
        print("=" * 60)
        sys.exit(1)


def get_local_ip():
    """Détecte l'IP locale sur le réseau OptiTrack."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((SERVER_IP, 1510))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "inconnue"


def run_test(use_multicast: bool):
    """
    Lance un test de connexion NatNet.

    Parameters
    ----------
    use_multicast : bool
        True  → multicast (défaut Motive, port 239.255.42.99:1511)
        False → unicast   (point-à-point, port SERVER_IP:1511)
    """
    NatNetClient = check_natnet_client_available()

    mode_str = "MULTICAST" if use_multicast else "UNICAST"
    print(f"\n{'='*60}")
    print(f"  Test {mode_str}")
    print(f"  Server IP : {SERVER_IP}")
    print(f"  Client IP : {CLIENT_IP} (IP locale détectée : {get_local_ip()})")
    print(f"{'='*60}")

    # Compteurs partagés entre threads
    frame_count = [0]
    rigid_bodies_seen = {}   # name → dernière position
    lock = threading.Lock()

    # ── Callbacks ─────────────────────────────────────────────────
    def receive_rigid_body_frame(new_id, position, rotation):
        """Appelé pour chaque rigid body dans chaque frame."""
        with lock:
            frame_count[0] += 1
            name = str(new_id)
            rigid_bodies_seen[name] = {
                'pos': position,
                'rot': rotation,
                'last_frame': frame_count[0],
            }

    def receive_new_frame(data_dict):
        """Appelé à chaque nouvelle frame (toutes données)."""
        # On compte ici seulement si aucun rigid body n'est défini
        if 'rigid_bodies' not in data_dict:
            with lock:
                frame_count[0] += 1

    # ── Créer le client ───────────────────────────────────────────
    client = NatNetClient.NatNetClient()
    client.set_client_address(CLIENT_IP)
    client.set_server_address(SERVER_IP)
    client.set_use_multicast(use_multicast)

    # SDK 4.4 : new_frame_with_data_listener est le callback principal,
    # il reçoit un objet MoCapData complet avec tous les rigid bodies.
    def receive_frame_with_data(data):
        with lock:
            frame_count[0] += 1
            try:
                for rb in data.rigid_body_data.rigid_body_list:
                    name = str(rb.id_num)
                    rigid_bodies_seen[name] = {
                        'pos': (rb.pos[0], rb.pos[1], rb.pos[2]),
                        'rot': (rb.rot[0], rb.rot[1], rb.rot[2], rb.rot[3]),
                        'last_frame': frame_count[0],
                    }
            except Exception:
                pass

    if hasattr(client, 'new_frame_with_data_listener'):
        client.new_frame_with_data_listener = receive_frame_with_data
    if hasattr(client, 'new_frame_listener'):
        client.new_frame_listener = receive_new_frame
    if hasattr(client, 'rigid_body_listener'):
        client.rigid_body_listener = receive_rigid_body_frame

    # ── Lancer la connexion ───────────────────────────────────────
    print(f"[{mode_str}] Connexion en cours...")
    # SDK 4.4 : run() attend un argument 'thread_option'
    # SDK 3.x : run() sans argument
    try:
        is_running = client.run("threaded")
    except TypeError:
        is_running = client.run()

    if not is_running:
        print(f"[{mode_str}] ÉCHEC : impossible de démarrer le client NatNet.")
        print(f"           → Vérifie que le port UDP 1510/1511 n'est pas bloqué")
        print(f"             par le firewall Windows.")
        print()
        print("  Pour ouvrir les ports dans PowerShell (admin) :")
        print("  New-NetFirewallRule -DisplayName 'NatNet in' -Direction Inbound "
              "-Protocol UDP -LocalPort 1510,1511 -Action Allow")
        return False

    print(f"[{mode_str}] Client démarré — écoute pendant {DURATION_S}s...")

    # ── Écouter pendant DURATION_S secondes ───────────────────────
    t_start = time.time()
    last_report = -1
    while time.time() - t_start < DURATION_S:
        elapsed = int(time.time() - t_start)
        with lock:
            fc = frame_count[0]
            rb = dict(rigid_bodies_seen)

        if elapsed != last_report:
            last_report = elapsed
            if fc == 0:
                print(f"  [{elapsed:2d}s] En attente de données... "
                      f"(0 frames reçues)")
            else:
                rb_names = list(rb.keys()) if rb else ["(aucun rigid body)"]
                print(f"  [{elapsed:2d}s] {fc} frames reçues — "
                      f"Rigid bodies : {rb_names}")

        time.sleep(0.2)

    # ── Résumé ────────────────────────────────────────────────────
    client.shutdown()

    with lock:
        fc = frame_count[0]
        rb = dict(rigid_bodies_seen)

    print(f"\n[{mode_str}] Résultat après {DURATION_S}s :")
    if fc == 0:
        print(f"  ✗ AUCUNE frame reçue")
        print(f"    Causes possibles :")
        if use_multicast:
            print(f"    - Multicast bloqué par le switch/routeur réseau")
            print(f"    - Motive streame en unicast → réessayer en UNICAST")
        else:
            print(f"    - IP du serveur incorrecte ({SERVER_IP})")
            print(f"    - Firewall bloque UDP 1510/1511")
            print(f"    - Motive ne streame pas (vérifier le panneau Streaming)")
        return False
    else:
        fps_approx = fc / DURATION_S
        print(f"  ✓ {fc} frames reçues (~{fps_approx:.0f} fps)")
        if rb:
            print(f"  ✓ {len(rb)} rigid body(ies) détecté(s) :")
            for name, data in rb.items():
                x, y, z = data['pos']
                qx, qy, qz, qw = data['rot']
                print(f"      ID {name} : pos=({x:.3f}, {y:.3f}, {z:.3f})  "
                      f"quat=({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})")
        else:
            print(f"  ⚠ Frames reçues mais AUCUN rigid body visible")
            print(f"    → Vérifie que le rigid body est bien défini et activé "
                  f"dans Motive")
        return True


def main():
    print("=" * 60)
    print("  test_natnet.py — Diagnostic connexion OptiTrack")
    print("=" * 60)

    # Test multicast en premier (défaut Motive)
    ok_multicast = run_test(use_multicast=True)

    if not ok_multicast:
        print("\n→ Multicast a échoué, test en UNICAST...")
        ok_unicast = run_test(use_multicast=False)
        if not ok_unicast:
            print("\n" + "=" * 60)
            print("  DIAGNOSTIC FINAL : Aucune connexion établie")
            print()
            print("  Checklist :")
            print("  [ ] Motive est ouvert et streaming est activé")
            print("      (View → Data Streaming → Broadcast Frame Data = ON)")
            print("  [ ] Local Interface dans Motive = IP de la carte réseau")
            print("      reliée au réseau 192.168.0.x (pas 127.0.0.1)")
            print("  [ ] Ce PC et le PC Motive sont sur le même sous-réseau")
            print(f"     (ping {SERVER_IP} fonctionne ?)")
            print("  [ ] Firewall Windows désactivé ou ports 1510/1511 ouverts")
            print("  [ ] Une seule interface réseau active sur ce PC")
            print("      (désactiver le WiFi si Ethernet est utilisé)")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("  → UNICAST fonctionne !")
            print("  Dans Motive : Data Streaming → Transmission Type = Unicast")
            print("  Le script principal devrait fonctionner avec unicast.")
            print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("  → MULTICAST fonctionne !")
        print("  La connexion OptiTrack est opérationnelle.")
        print("=" * 60)


if __name__ == "__main__":
    main()