import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class FridgeDRSim:
    def __init__(self, fleet_size=1000000, batt_capacity_wh=500, batt_power_w=500):
        self.fridge_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'fridge_power.csv'))
        self.n_fridges = 10000
        self.hourly_load_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'isone_hourly.csv'))
        self.hourly_load_data['timestamp'] = pd.to_datetime(self.hourly_load_data['timestamp'])
        self.hourly_load_data.set_index('timestamp', inplace=True)

        self.batt_capacity_wh = batt_capacity_wh
        self.batt_power_w = batt_power_w
        self.fleet_size = fleet_size

        self.aggregate_load = self.create_virtual_fleet()
        

    def create_virtual_fleet(self):
        """
        Simulates a fleet using the 'Tile and Slice' method to eliminate
        edge spikes and harmonics.
        """
        source_matrix = self.fridge_data.values
        n_source_cols = source_matrix.shape[1]
        n_timesteps = source_matrix.shape[0]  # e.g., 1440 minutes
        
        # 1. Tile the data 3 times (Day-1, Day, Day+1)
        # This creates a buffer so we can slide the window without wrapping abruptly
        # Shape becomes (4320, n_source_cols)
        tiled_source = np.tile(source_matrix, (3, 1))
        
        # 2. Assign profiles
        assigned_profiles = np.random.randint(0, n_source_cols, self.n_fridges)
        
        # 3. Generate Random Start Offsets (The "Scramble")
        # We pick a random start minute anywhere within the first 24 hours (0 to 1440)
        start_offsets = np.random.randint(0, n_timesteps, self.n_fridges)
        
        # 4. Aggregate
        total_fleet_load = np.zeros(n_timesteps)
        
        # Vectorized accumulation
        # We iterate through the simulated units (or batches of them for speed)
        # For a few thousand units, a simple loop is actually quite fast with Numpy slicing
        for i in range(self.n_fridges):
            src_idx = assigned_profiles[i]
            offset = start_offsets[i]
            
            # SLICE the continuous window from the tiled matrix
            # We take 1440 minutes starting from the random offset
            # This guarantees a smooth, unbroken profile for every fridge
            fridge_profile = tiled_source[offset : offset + n_timesteps, src_idx]
            
            total_fleet_load += fridge_profile
                
        return total_fleet_load

    def load_grid_data(self, date):
        dstart = pd.to_datetime(date + ' 00:00')
        dend = dstart + pd.Timedelta(days=1, minutes=1)

        day_load_hourly = self.hourly_load_data.loc[dstart:dend]
        if len(day_load_hourly) != 25:
            print(f"Error: {date} has {len(day_load_hourly)} hours of data, expected 25")
        day_load_hourly.index = pd.date_range(dstart, dend, freq='h')

        df_grid_min = day_load_hourly.resample('1min').asfreq()
        df_grid_min['Load_MW'] = df_grid_min['Load_MW'].interpolate(method='cubic')

        return df_grid_min

    def find_peak_shaving_window(self, grid_load_series):
        """
        Finds the optimal discharge window that IS GUARANTEED to include 
        the absolute maximum peak load of the day.
        """
        # 1. Calculate Duration (in minutes)
        duration_hours = self.batt_capacity_wh / self.batt_power_w
        duration_minutes = int(duration_hours * 60)
        
        peak_idx = grid_load_series.argmax()
        
        # Clamp to ensure we don't go off the edges of the day
        min_start_idx = max(0, peak_idx - duration_minutes + 1)
        max_start_idx = min(len(grid_load_series) - duration_minutes, peak_idx)
        
        # 4. Search within valid constraints for the "Heaviest" Window
        # We only look at windows that touch the peak.
        best_avg_load = -1
        best_start_idx = min_start_idx
        
        # Iterate through possible alignments
        for start in range(min_start_idx, max_start_idx + 1):
            end = start + duration_minutes
            
            # Calculate average load of this candidate window
            current_avg = grid_load_series.iloc[start:end].mean()
            
            if current_avg > best_avg_load:
                best_avg_load = current_avg
                best_start_idx = start
                
        # 5. Convert back to hours for the simulation input
        best_end_idx = best_start_idx + duration_minutes
        
        start_hour = best_start_idx / 60
        end_hour = best_end_idx / 60
        
        return start_hour, end_hour

    def simulate_demand_response(self,
        event_start_hour=17, 
        event_end_hour=19,
        recharge_window_hours=8
    ):
        """
        Simulates a battery-backed fleet responding to a peak event.
        
        Args:
            event_start/end_hour (float): The DR window (e.g., 17 = 5 PM)
            recharge_window_hours (float): How long the fleet takes to recharge
            
        Returns:
            pd.DataFrame with columns: ['Baseline', 'With_DR', 'Battery_SoC', 'Action']
        """
        
        baseline_load_kw = self.aggregate_load / 1000

        # 1. Scale Battery Specs to Fleet Level (convert to kW and kWh)
        fleet_capacity_kwh = (self.n_fridges * self.batt_capacity_wh) / 1000
        fleet_max_power_kw = (self.n_fridges * self.batt_power_w) / 1000
        
        # Time setup
        minutes = len(baseline_load_kw)
        time_indices = np.arange(minutes)
        
        # Convert hours to minute indices
        start_idx = int(event_start_hour * 60)
        end_idx = int(event_end_hour * 60)
        recharge_end_idx = min(minutes, int(end_idx + (recharge_window_hours * 60)))
        
        # Initialize arrays
        new_load_kw = baseline_load_kw.copy()
        soc_kwh = np.full(minutes, fleet_capacity_kwh) # Start full
        action_log = np.zeros(minutes) # For visualization (positive=discharge)
        
        # ==========================================
        # SIMULATION LOOP
        # ==========================================
        
        current_soc = fleet_capacity_kwh
        
        for t in range(minutes):
            
            # A. DISCHARGE LOGIC (During Peak Window)
            if start_idx <= t < end_idx:
                if current_soc > 0:
                    # 1. How much do we WANT to discharge? (Cap at max power)
                    target_discharge = fleet_max_power_kw
                    
                    # 2. How much energy do we HAVE? (kWh -> kW conversion)
                    # Energy = Power * Time. So Power = Energy / (1/60 hours)
                    max_energy_discharge_kw = current_soc * 60 
                    
                    # 3. How much load is ACTUALLY there? (Can't discharge more than load)
                    # (Assuming no grid backfeeding allowed)
                    real_load_limit = new_load_kw[t]
                    
                    # The actual discharge is the minimum of all constraints
                    actual_discharge_kw = min(target_discharge, max_energy_discharge_kw, real_load_limit)
                    
                    # Apply
                    new_load_kw[t] -= actual_discharge_kw
                    current_soc -= (actual_discharge_kw / 60) # Convert kW back to kWh
                    action_log[t] = -actual_discharge_kw
                    
            # B. RECHARGE LOGIC (After Peak Window)
            elif end_idx + 60 <= t < recharge_end_idx:
                # Calculate how much energy is missing
                missing_energy = fleet_capacity_kwh - current_soc
                
                if missing_energy > 0:
                    # Simple linear recharge distribution
                    # (In reality, this follows a CC/CV curve, but linear is fine for grid sim)
                    remaining_minutes = recharge_end_idx - t
                    recharge_rate_kw = (missing_energy * 60) / remaining_minutes
                    
                    # Cap recharge rate at max power (batteries limit charge speed too)
                    recharge_rate_kw = min(recharge_rate_kw, fleet_max_power_kw)
                    
                    new_load_kw[t] += recharge_rate_kw
                    current_soc += (recharge_rate_kw / 60)
                    action_log[t] = recharge_rate_kw

            # Store State of Charge
            soc_kwh[t] = current_soc

        # Package results
        return pd.DataFrame({
            'Baseline': baseline_load_kw,
            'With_DR': new_load_kw,
            'Battery_SoC': soc_kwh,
            'Battery_Action': action_log
        })

    def evaluate_grid_impact(self, dr_simulation, grid_data):
        fleet_delta_kw = dr_simulation['With_DR'] - dr_simulation['Baseline']
        scaling_factor = self.fleet_size / self.n_fridges
        fleet_delta_mw_scaled = (fleet_delta_kw / 1000) * scaling_factor
        
        sim_len = len(fleet_delta_mw_scaled)
        original_grid_profile = grid_data['Load_MW'].values[:sim_len]
        new_grid_profile = original_grid_profile + fleet_delta_mw_scaled
        
        return original_grid_profile, new_grid_profile

    def run(self, date=None):
        grid_data = self.load_grid_data(date)

        peak_start, peak_end = self.find_peak_shaving_window(grid_data['Load_MW'])
        
        dr_simulation = self.simulate_demand_response(
            event_start_hour=peak_start, 
            event_end_hour=peak_end,
            recharge_window_hours=4
        )

        original_grid_profile, new_grid_profile = self.evaluate_grid_impact(dr_simulation, grid_data)

        original_peak = np.max(original_grid_profile)
        new_peak = np.max(new_grid_profile)
        stats = {
            'original_peak': original_peak,
            'new_peak': new_peak,
            'mw_reduction': original_peak - new_peak,
        }

        ret = {
            'stats': stats,
            'original_grid_profile': original_grid_profile,
            'new_grid_profile': new_grid_profile,
            'grid_data': grid_data,
            'sim_length': len(self.aggregate_load),
            'date' : date,
            'dr_start_hour' : peak_start,
            'dr_end_hour' : peak_end
        }

        return ret
        