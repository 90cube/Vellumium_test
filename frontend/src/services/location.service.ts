
import { Injectable, inject, signal } from '@angular/core';
import { SupabaseService } from './supabase.service';
import { AuthService } from './auth.service';
import { Location } from '../models/location.model';

@Injectable({ providedIn: 'root' })
export class LocationService {
  private supabase = inject(SupabaseService).supabase;
  private authService = inject(AuthService);

  /** All locations for the currently loaded project */
  readonly locations = signal<Location[]>([]);

  /** Currently selected location for editing */
  readonly selectedLocation = signal<Location | null>(null);

  // ── CRUD ────────────────────────────────────────────────

  async loadLocations(projectId: string): Promise<void> {
    const { data, error } = await this.supabase
      .from('locations')
      .select('*')
      .eq('project_id', projectId)
      .order('sort_order', { ascending: true });

    if (error) {
      console.error('loadLocations error:', error);
      return;
    }

    const locations: Location[] = (data || []).map((row: any) => ({
      id: row.id,
      project_id: row.project_id,
      name: row.name,
      thumbnail_url: row.thumbnail_url,
      sliders: row.sliders ?? {},
      prompt: row.prompt ?? '',
      sort_order: row.sort_order ?? 0,
      created_at: row.created_at,
    }));

    this.locations.set(locations);
  }

  async createLocation(projectId: string, name: string): Promise<Location | null> {
    const currentLocations = this.locations();
    const maxOrder = currentLocations.length > 0
      ? Math.max(...currentLocations.map(l => l.sort_order))
      : -1;

    const { data, error } = await this.supabase
      .from('locations')
      .insert({
        project_id: projectId,
        name,
        sliders: {},
        prompt: '',
        sort_order: maxOrder + 1,
      })
      .select()
      .single();

    if (error) {
      console.error('createLocation error:', error);
      return null;
    }

    const location: Location = {
      id: data.id,
      project_id: data.project_id,
      name: data.name,
      thumbnail_url: data.thumbnail_url,
      sliders: data.sliders ?? {},
      prompt: data.prompt ?? '',
      sort_order: data.sort_order ?? 0,
      created_at: data.created_at,
    };

    this.locations.update(prev => [...prev, location]);
    return location;
  }

  async updateLocation(id: string, updates: Partial<Location>): Promise<boolean> {
    const { error } = await this.supabase
      .from('locations')
      .update(updates)
      .eq('id', id);

    if (error) {
      console.error('updateLocation error:', error);
      return false;
    }

    this.locations.update(prev =>
      prev.map(loc => loc.id === id ? { ...loc, ...updates } : loc)
    );

    // Also update selectedLocation if it matches
    const selected = this.selectedLocation();
    if (selected && selected.id === id) {
      this.selectedLocation.set({ ...selected, ...updates });
    }

    return true;
  }

  async deleteLocation(id: string): Promise<boolean> {
    const { error } = await this.supabase
      .from('locations')
      .delete()
      .eq('id', id);

    if (error) {
      console.error('deleteLocation error:', error);
      return false;
    }

    this.locations.update(prev => prev.filter(loc => loc.id !== id));

    // Clear selection if deleted location was selected
    const selected = this.selectedLocation();
    if (selected && selected.id === id) {
      this.selectedLocation.set(null);
    }

    return true;
  }
}
