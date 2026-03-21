import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
import time

URI = 'radio://0/80/2M/E7E7E7E7E7'

def main():
    cflib.crtp.init_drivers()
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        lc = LogConfig('Att', period_in_ms=100)
        lc.add_variable('stateEstimate.pitch', 'float')
        scf.cf.log.add_config(lc)

        def cb(ts, data, _):
            print(f"pitch = {data['stateEstimate.pitch']:+.2f} deg")

        lc.data_received_cb.add_callback(cb)
        lc.start()
        time.sleep(10)
        lc.stop()

main()