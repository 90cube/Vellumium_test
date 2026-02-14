
import { Injectable, signal, computed, inject } from '@angular/core';
import { SupabaseService } from './supabase.service';
import type { User } from '@supabase/supabase-js';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private supabaseService = inject(SupabaseService);
  private supabase = this.supabaseService.supabase;

  readonly currentUser = signal<User | null>(null);
  readonly isLoggedIn = computed(() => this.currentUser() !== null);
  readonly authError = signal<string | null>(null);
  readonly authLoading = signal(false);

  constructor() {
    // Check current session on init
    this.supabase.auth.getSession().then(({ data: { session } }) => {
      this.currentUser.set(session?.user ?? null);
    });

    // Listen for auth state changes
    this.supabase.auth.onAuthStateChange((_event, session) => {
      this.currentUser.set(session?.user ?? null);
    });
  }

  async signInWithEmail(email: string, password: string): Promise<boolean> {
    this.authError.set(null);
    this.authLoading.set(true);
    try {
      const { data, error } = await this.supabase.auth.signInWithPassword({ email, password });
      if (error) {
        this.authError.set(error.message);
        return false;
      }
      this.currentUser.set(data.user);
      return true;
    } catch (e: any) {
      this.authError.set(e.message || 'Sign in failed');
      return false;
    } finally {
      this.authLoading.set(false);
    }
  }

  async signUpWithEmail(email: string, password: string): Promise<boolean> {
    this.authError.set(null);
    this.authLoading.set(true);
    try {
      const { data, error } = await this.supabase.auth.signUp({ email, password });
      if (error) {
        this.authError.set(error.message);
        return false;
      }
      // If email confirmation is required, user won't be immediately logged in
      if (data.user && !data.user.confirmed_at) {
        this.authError.set('Check your email for a confirmation link.');
        return false;
      }
      this.currentUser.set(data.user);
      return true;
    } catch (e: any) {
      this.authError.set(e.message || 'Sign up failed');
      return false;
    } finally {
      this.authLoading.set(false);
    }
  }

  async signInWithGoogle(): Promise<void> {
    this.authError.set(null);
    const { error } = await this.supabase.auth.signInWithOAuth({ provider: 'google' });
    if (error) {
      this.authError.set(error.message);
    }
    // OAuth redirects the page, so no further handling needed here
  }

  async signOut(): Promise<void> {
    this.authError.set(null);
    const { error } = await this.supabase.auth.signOut();
    if (error) {
      this.authError.set(error.message);
    }
    this.currentUser.set(null);
  }
}
