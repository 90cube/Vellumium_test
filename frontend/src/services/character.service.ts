
import { Injectable, inject, signal } from '@angular/core';
import { SupabaseService } from './supabase.service';
import { AuthService } from './auth.service';
import { Character } from '../models/character.model';

@Injectable({
  providedIn: 'root'
})
export class CharacterService {
  private supabase = inject(SupabaseService).supabase;
  private authService = inject(AuthService);

  readonly characters = signal<Character[]>([]);
  readonly selectedCharacter = signal<Character | null>(null);

  async loadCharacters(projectId: string): Promise<void> {
    const { data, error } = await this.supabase
      .from('characters')
      .select('*')
      .eq('project_id', projectId)
      .order('sort_order', { ascending: true });

    if (error) {
      console.error('loadCharacters error:', error);
      return;
    }

    const chars: Character[] = (data || []).map((row: any) => ({
      id: row.id,
      project_id: row.project_id,
      name: row.name,
      thumbnail_url: row.thumbnail_url,
      sliders: row.sliders ?? {},
      prompt: row.prompt ?? '',
      sort_order: row.sort_order ?? 0,
      created_at: row.created_at,
    }));

    this.characters.set(chars);
  }

  async createCharacter(projectId: string, name: string): Promise<Character | null> {
    const currentChars = this.characters();
    const maxOrder = currentChars.length > 0
      ? Math.max(...currentChars.map(c => c.sort_order))
      : -1;

    const { data, error } = await this.supabase
      .from('characters')
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
      console.error('createCharacter error:', error);
      return null;
    }

    const character: Character = {
      id: data.id,
      project_id: data.project_id,
      name: data.name,
      thumbnail_url: data.thumbnail_url,
      sliders: data.sliders ?? {},
      prompt: data.prompt ?? '',
      sort_order: data.sort_order ?? 0,
      created_at: data.created_at,
    };

    this.characters.update(prev => [...prev, character]);
    return character;
  }

  async updateCharacter(id: string, updates: Partial<Character>): Promise<Character | null> {
    const { data, error } = await this.supabase
      .from('characters')
      .update(updates)
      .eq('id', id)
      .select()
      .single();

    if (error) {
      console.error('updateCharacter error:', error);
      return null;
    }

    const updated: Character = {
      id: data.id,
      project_id: data.project_id,
      name: data.name,
      thumbnail_url: data.thumbnail_url,
      sliders: data.sliders ?? {},
      prompt: data.prompt ?? '',
      sort_order: data.sort_order ?? 0,
      created_at: data.created_at,
    };

    this.characters.update(prev => prev.map(c => c.id === id ? updated : c));

    // Also update selectedCharacter if it matches
    if (this.selectedCharacter()?.id === id) {
      this.selectedCharacter.set(updated);
    }

    return updated;
  }

  async deleteCharacter(id: string): Promise<boolean> {
    const { error } = await this.supabase
      .from('characters')
      .delete()
      .eq('id', id);

    if (error) {
      console.error('deleteCharacter error:', error);
      return false;
    }

    this.characters.update(prev => prev.filter(c => c.id !== id));

    if (this.selectedCharacter()?.id === id) {
      this.selectedCharacter.set(null);
    }

    return true;
  }
}
