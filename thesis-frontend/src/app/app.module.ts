import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { MatDialogModule } from '@angular/material/dialog';
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
import { MatGridListModule } from '@angular/material/grid-list';
import { WordCloudComponentComponent } from './word-cloud-component/word-cloud-component.component';
import { WordReassignComponentComponent } from './word-reassign-component/word-reassign-component.component';
import { FeatureWeightingComponentComponent } from './feature-weighting-component/feature-weighting-component.component';
import { LandingPopupComponent } from './landing-popup/landing-popup.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { MatButtonModule } from '@angular/material/button';
import {  MatCardModule } from '@angular/material/card';

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
    WordReassignComponentComponent,
    FeatureWeightingComponentComponent,
    LandingPopupComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    FormsModule,
    HttpClientModule,
    NgChartsModule,
    MatDialogModule,
    MatGridListModule,
    NgxSliderModule,
    NgxSpinnerModule,
    BrowserAnimationsModule,
    MatButtonModule,
    MatCardModule
  ],
  providers: [],
  entryComponents: [AppComponent,LandingPopupComponent,ScatterComponentComponent,FacetExplanationComponentComponent],
  bootstrap: [AppComponent]
})
export class AppModule { }
