import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import {NgChartsModule} from 'ng2-charts';
import { FormsModule } from '@angular/forms';
import { ScatterComponentComponent } from './scatter-component/scatter-component.component';
import { GlobalExplanationsComponent } from './global-explanations/global-explanations.component';
import { LocalExplanationComponent } from './local-explanation/local-explanation.component';
import { HighlightWordsPipe } from './highlight-words.pipe';

@NgModule({
  declarations: [
    AppComponent,
    ScatterComponentComponent,
    GlobalExplanationsComponent,
    LocalExplanationComponent,
    HighlightWordsPipe
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    FormsModule,
    HttpClientModule,
    NgChartsModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
