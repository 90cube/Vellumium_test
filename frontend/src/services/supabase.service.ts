
import { Injectable } from '@angular/core';
import { createClient, SupabaseClient } from '@supabase/supabase-js';

const SUPABASE_URL = 'https://lxhtjxonftnzlknnpzza.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx4aHRqeG9uZnRuemxrbm5wenphIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzA1MzEyODEsImV4cCI6MjA4NjEwNzI4MX0.3uFfCTJzUQhgkRKfgrluIc0TFcJJ3wF5F6i4bIP9hzY';

@Injectable({
  providedIn: 'root'
})
export class SupabaseService {
  readonly supabase: SupabaseClient;

  constructor() {
    this.supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
  }
}
