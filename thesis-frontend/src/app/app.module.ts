import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { MatDialogModule } from '@angular/material/dialog';
import {NgChartsModule} from 'ng2-charts';
import { FormsModule } from '@angular/forms';
import { ScatterComponentComponent } from './scatter-component/scatter-component.component';
import { GlobalExplanationsComponent } from './global-explanations/global-explanations.component';
import { LocalExplanationComponent } from './local-explanation/local-explanation.component';
import { HighlightWordsPipe } from './highlight-words.pipe';
import { ArticleExplanationComponent } from './article-explanation/article-explanation.component';
import { HomeComponentComponent } from './home-component/home-component.component';
import { FacetExplanationComponentComponent } from './facet-explanation-component/facet-explanation-component.component';
import { MatGridListModule } from '@angular/material/grid-list';



@NgModule({
  declarations: [
    AppComponent,
    ScatterComponentComponent,
    GlobalExplanationsComponent,
    LocalExplanationComponent,
    HighlightWordsPipe,
    ArticleExplanationComponent,
    HomeComponentComponent,
    FacetExplanationComponentComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    FormsModule,
    HttpClientModule,
    NgChartsModule,
    MatDialogModule,
    MatGridListModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
