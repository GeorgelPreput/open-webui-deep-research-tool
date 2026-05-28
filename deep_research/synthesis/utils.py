def get_synthesis_model(ctx):
    return ctx.valves.models.synthesis_model or ctx.valves.models.research_model
