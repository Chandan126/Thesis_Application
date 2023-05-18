import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { BrowserModule } from '@angular/platform-browser';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { MaterialModule } from './material.module';
import { HttpClientModule } from '@angular/common/http';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { NgxSliderModule } from '@angular-slider/ngx-slider';
import {NgChartsModule} from 'ng2-charts';
import { NgxSpinnerModule } from 'ngx-spinner';
import { FormsModule } from '@angular/forms';
import { ScatterComponentComponent } from './scatter-component/scatter-component.component';
import { GlobalExplanationsComponent } from './global-explanations/global-explanations.component';
import { LocalExplanationComponent } from './local-explanation/local-explanation.component';
import { HighlightWordsPipe } from './highlight-words.pipe';
import { ArticleExplanationComponent } from './article-explanation/article-explanation.component';
import { HomeComponentComponent } from './home-component/home-component.component';
import { FacetExplanationComponentComponent } from './facet-explanation-component/facet-explanation-component.component';
import { WordCloudComponentComponent } from './word-cloud-component/word-cloud-component.component';
import { FeatureWeightingComponentComponent } from './feature-weighting-component/feature-weighting-component.component';
import { LandingPopupComponent } from './landing-popup/landing-popup.component';
import { PlotlyComponentComponent } from './plotly-component/plotly-component.component';

import * as PlotlyJS from 'plotly.js-dist-min';
import { PlotlyModule } from 'angular-plotly.js';

PlotlyModule.plotlyjs = PlotlyJS;

@NgModule({
  declarations: [
    AppComponent,
    ScatterComponentComponent,
    GlobalExplanationsComponent,
    LocalExplanationComponent,
    HighlightWordsPipe,
    ArticleExplanationComponent,
    HomeComponentComponent,
    FacetExplanationComponentComponent,
    WordCloudComponentComponent,
    FeatureWeightingComponentComponent,
    LandingPopupComponent,
    PlotlyComponentComponent
  ],
  imports: [
    BrowserModule,
    BrowserAnimationsModule,
    AppRoutingModule,
    MaterialModule,
    FormsModule,
    HttpClientModule,
    NgChartsModule,
    NgxSliderModule,
    NgxSpinnerModule,
    CommonModule, 
    PlotlyModule  ],
  providers: [],
  entryComponents: [AppComponent,LandingPopupComponent,ScatterComponentComponent,FacetExplanationComponentComponent],
  bootstrap: [AppComponent]
})
export class AppModule { }
